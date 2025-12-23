import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math


class RNN_Model(nn.Module):
    # graph_infor提供图邻接矩阵信息， params中的graph_choose决定选择哪些图结构，且列表中的参数模块根据graph_choose排序
    def __init__(self, params, graph_infor):
        super(RNN_Model, self).__init__()
        self.params = params
        self.W_1, self.W_2 = nn.ModuleList(), nn.ModuleList()
        self.graph_mat = []
        for gr in params.graph_choose.keys():
            if params.graph_choose[gr]:
                sqr_D_head = torch.tensor(graph_infor[gr]['sqr_D_head'], device=self.params.device)
                A_head = torch.tensor(graph_infor[gr]['A_head'], device=self.params.device)
                self.W_1.append(nn.Linear(self.params.cov_dim + 1, self.params.lstm_hidden_dim, bias=False))
                self.W_2.append(nn.Linear(self.params.lstm_hidden_dim, self.params.lstm_hidden_dim, bias=False))
                self.graph_mat.append(torch.matmul(torch.matmul(sqr_D_head, A_head), sqr_D_head).to(torch.float32))

        self.embedding = nn.Embedding(params.num_series, params.embedding_dim)
        self.lstm = nn.LSTM(input_size=params.lstm_hidden_dim+params.embedding_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.mu = nn.ModuleList()
        for series in range(self.params.num_series):
            self.mu.append(nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1))
        self.sigma = nn.ModuleList()
        for series in range(self.params.num_series):
            self.sigma.append(nn.Linear(params.lstm_hidden_dim * params.lstm_layers, 1))

    def forward(self, train_x, hidden, cell):
        """
       :param train_x: [B, Nodes_num, 1+cov_dim+1]
       :param hidden: [lstm_layers*directions, batch_size*Nodes_num, lstm_hidden_dim]
       :param cell: [lstm_layers*directions, batch_size*Nodes_num, lstm_hidden_dim]
       :return:hidden, cell, mu[B, Nodes_num]
       """
        batch_size = train_x.shape[0]
        graph_input = train_x[:, :, :-1]
        lstm_input = torch.zeros((train_x.shape[0], train_x.shape[1], self.params.lstm_hidden_dim))
        # 对各关联均进行原始特征的图卷积，再相加
        for i in range(len(self.graph_mat)):
            graph_feature = F.relu(torch.matmul(self.graph_mat[i], self.W_1[i](graph_input)))
            graph_feature = F.relu(torch.matmul(self.graph_mat[i], self.W_2[i](graph_feature)))
            lstm_input += graph_feature

        embedding = self.embedding(train_x[:, :, -1].to(torch.int64))
        lstm_input = torch.cat([lstm_input, embedding], dim=-1)

        lstm_input = torch.reshape(lstm_input, (-1, self.params.lstm_hidden_dim+self.params.embedding_dim)).unsqueeze(0)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        hidden_permute = hidden_permute.reshape(batch_size, self.params.num_series, -1)  # [B, N, D]
        mu, sigma = [], []
        for series in range(self.params.num_series):
            mu.append(self.mu[series](hidden_permute[:, series]).reshape(batch_size))
            sigma.append(F.softplus(self.sigma[series](hidden_permute[:, series])).reshape(batch_size))
        mu, sigma = torch.stack(mu, dim=1), torch.stack(sigma, dim=1)

        return mu, sigma, hidden, cell

    def init_hidden(self, batch_size):
        return torch.zeros(self.params.lstm_layers, batch_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, batch_size):
        return torch.zeros(self.params.lstm_layers, batch_size, self.params.lstm_hidden_dim, device=self.params.device)

    def test(self, val_input, hidden, cell, sampling=False):
        batch_size = val_input.shape[1]
        if sampling:
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.for_days, self.params.num_series,
                                  device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.for_days):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(
                        val_input[self.params.lag_days - 1 + t], decoder_hidden, decoder_cell)
                    gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                    pred = gaussian.sample()
                    samples[j, :, t, :] = pred
                    if t < (self.params.for_days - 1):
                        val_input[self.params.lag_days + t, :, :, 0] = pred

            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma

        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(batch_size, self.params.for_days, self.params.num_series, device=self.params.device)
            sample_sigma = torch.zeros(batch_size, self.params.for_days, self.params.num_series,
                                       device=self.params.device)
            for t in range(self.params.for_days):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(
                    val_input[self.params.lag_days - 1 + t], decoder_hidden, decoder_cell)
                sample_mu[:, t, :] = mu_de.clone()
                sample_sigma[:, t, :] = sigma_de.clone()
                if t < (self.params.for_days - 1):
                    val_input[self.params.lag_days + t, :, :, 0] = mu_de
            return sample_mu, sample_sigma


def loss_fn_gassion(mu: Variable, sigma: Variable, labels: Variable):
    """
    Compute using gaussian the log-likelihood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    """
    distribution = torch.distributions.normal.Normal(mu, sigma)
    likelihood = distribution.log_prob(labels)
    return -torch.mean(likelihood)


def loss_fn_regression(for_para, label, for_days, lamda):
    loss = F.smooth_l1_loss(for_para['mu'], label)
    return torch.mean(loss)


def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    if not relative:
        diff = torch.sum(torch.abs(mu - labels)).item()
        summation = mu.shape[0] * mu.shape[1] * mu.shape[2]
        return [diff, summation]
    else:
        diff = torch.sum(torch.abs(mu - labels) / torch.abs(labels)).item()
        summation = torch.sum(torch.abs(labels)).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative=False):
    if not relative:
        diff = torch.sum(torch.mul(torch.abs(mu - labels), torch.abs(mu - labels))).item()
        summation = mu.shape[0] * mu.shape[1] * mu.shape[2]
        return [diff, summation, summation]
    else:
        diff = torch.sum(torch.mul(torch.abs(mu - labels), torch.abs(mu - labels)) / torch.mul(labels, labels)).item()
        return [diff, torch.sum(torch.abs(labels)).item(), (mu.shape[0] * mu.shape[1]).item()]


def accuracy_ROU(rou: float, samples: torch.Tensor, labels: torch.Tensor, relative=False):
    numerator = 0
    denominator = 0
    pred_samples = samples.shape[0]
    for node in range(labels.shape[2]):
        for t in range(labels.shape[1]):
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, :, t, node], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t, node] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t, node] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t, node] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t, node]).item()
    if not relative:
        return [numerator, labels.shape[0] * labels.shape[1] * labels.shape[2]]
    else:
        return [numerator, denominator]

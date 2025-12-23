import logging
import argparse
import os
import utils
import torch
import my_model
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd


def load_test(data_dir):
    x_test = np.load(os.path.join(data_dir, 'test_x.npy'))  #
    covariates = np.load(os.path.join(data_dir, 'test_covariate.npy'))
    scalar = np.load(os.path.join(data_dir, 'scalar.npy'))
    Date = np.load(os.path.join(data_dir, 'test_Date.npy'), allow_pickle=True)
    return x_test, covariates, scalar, Date


def test_forcast(model, x_test_label, covariates, params, scalar, Date, points):
    sampling = params.sampling
    model.eval()

    with torch.no_grad():
        covariates = covariates[:, 1:, :]
        x_test_for = np.copy(x_test_label[:params.lag_days])
        for series in range(params.num_series):
            x_test_for[:, series] = (x_test_for[:, series] - scalar[series, 1]) / (
                        scalar[series, 0] - scalar[series, 1])
        while x_test_for.shape[0] < x_test_label.shape[0]:
            start_index = x_test_for.shape[0] - params.lag_days
            covariate = torch.tensor(covariates[:, start_index:start_index + params.window_days - 1]).to(torch.float32).to(
                params.device)
            x_input = np.zeros((params.window_days - 1, params.num_series))
            x_input[:params.lag_days] = x_test_for[-params.lag_days:]
            x_input = torch.tensor(x_input).to(torch.float32).to(params.device)  # x_input[T, N]
            model_input = []
            for series in range(params.num_series):
                train_x_sery = torch.zeros((params.window_days - 1, 1), dtype=torch.int64, device=params.device)
                train_x_sery[:] = series
                node_input = torch.cat([x_input[:, series].unsqueeze(1), covariate[series], train_x_sery], dim=1)  # [T, D]
                model_input.append(node_input)
            model_input = torch.stack(model_input, dim=1).unsqueeze(0)  # [1, T, N, D]

            model_input = model_input.permute(1, 0, 2, 3)
            batch_size = model_input.shape[1]
            hidden = model.init_hidden(batch_size * params.num_series)
            cell = model.init_cell(batch_size * params.num_series)

            # 获得要进行解码预测时的hidden, cell
            for t in range(params.lag_days - 1):
                mu, sigma, hidden, cell = model(model_input[t].clone(), hidden, cell)

            # 在预测时没进行多次采样，而是直接取mu
            if sampling:
                samples = torch.zeros(params.sample_times, batch_size, params.for_days,
                                      params.num_series, device=params.device)
                for j in range(params.sample_times):
                    decoder_hidden = hidden
                    decoder_cell = cell
                    for t in range(params.for_days):
                        mu_de, sigma_de, decoder_hidden, decoder_cell = model(
                            model_input[params.lag_days - 1 + t], decoder_hidden, decoder_cell)
                        gaussian = torch.distributions.normal.Normal(mu_de, sigma_de)
                        pred = gaussian.sample()
                        samples[j, :, t, :] = pred
                        if t < (params.for_days - 1):
                            model_input[params.lag_days + t, :, :, 0] = pred
                for_mu = torch.median(samples, dim=0)[0]  # [1, T, Nodes]

            else:
                decoder_hidden = hidden
                decoder_cell = cell
                for_mu = torch.zeros(batch_size, params.for_days, params.num_series, device=params.device)
                for_sigma = torch.zeros(batch_size, params.for_days, params.num_series, device=params.device)
                for t in range(params.for_days):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = model(
                        model_input[params.lag_days - 1 + t], decoder_hidden, decoder_cell)
                    for_mu[:, t, :] = mu_de.clone()
                    for_sigma[:, t, :] = sigma_de.clone()
                    if t < (params.for_days - 1):
                        model_input[params.lag_days + t, :, :, 0] = mu_de

            for_mu = for_mu.squeeze(0).data.cpu().numpy()
            x_test_for = np.concatenate((x_test_for, for_mu), axis=0)

        x_test_for = x_test_for[:x_test_label.shape[0]]  # [T, Nodes_num] 与label长度一致
        # 预测完成 反归一化
        for series in range(params.num_series):
            x_test_for[:, series] = x_test_for[:, series] * (scalar[series, 0] - scalar[series, 1]) + scalar[series, 1]
        plot_test(x_test_label, x_test_for, params, points)
        result_to_excel(x_test_label, x_test_for, params, Date, points)


def result_to_excel(x_test_label, x_test_for, params, Date, points):
    x_test_label = x_test_label[params.lag_days:]
    x_test_for = x_test_for[params.lag_days:]
    df = pd.DataFrame()
    df['Date'] = Date[-x_test_label.shape[0]:]
    for num in range(x_test_label.shape[1]):
        df[points[num]+'_true'] = x_test_label[:, num]
        df[points[num]+'_pre'] = x_test_for[:, num]
    df.to_excel(os.path.join(params.plot_test_dir, 'test_results.xlsx'), index=None)


def plot_test(x_test_label, x_test_for, params, points):
    # x_test_label, x_test_for [T, Nodes_num]
    plot_num = x_test_label.shape[1]
    x = np.arange(x_test_label.shape[0])
    for pic in range(plot_num):
        f = plt.figure(figsize=(8, 4), constrained_layout=True)
        ax = f.subplots(1, 1)
        ax.plot(x, np.array(x_test_label[:, pic]), color='r')
        ax.plot(x, np.array(x_test_for[:, pic]), color='b')
        ax.axvline(params.lag_days, color='g', linestyle='dashed')
        f.savefig(os.path.join(params.plot_test_dir, points[pic]+'_test_forecast.png'))


if __name__ == '__main__':
    logger = logging.getLogger('MYAR.FOR')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
    parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
    parser.add_argument('--best_epoch', default=82, help='the best epoch during training')  # 'best' or 'epoch_#'
    parser.add_argument('--sampling', default=False, help='Whether to sample during evaluation')
    args = parser.parse_args()

    #  选择哪些图结构信息
    graph_choose = {'DTW': True,
                    'Gauss': True,
                    'Partition': True,
                    'Line': True}

    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder)
    params = utils.Params(json_path)
    params.graph_choose = graph_choose
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_test_dir = os.path.join(model_dir, 'figures')
    params.best_model_path = os.path.join(model_dir, 'best_{}epoch.pth.tar'.format(args.best_epoch))

    with open(os.path.join(data_dir, 'Graph_info.pkl'), "rb") as tf:
        graph_infor = pickle.load(tf)
    # use GPU if available
    cuda_exist = torch.cuda.is_available()
    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = my_model.RNN_Model(params, graph_infor).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = my_model.RNN_Model(params, graph_infor)

    checkpoint = utils.load_checkpoint(params.best_model_path, model)

    utils.set_logger(os.path.join(model_dir, 'test.log'))
    logger.info('Loading the test_datasets...')

    x_test, covariates, scalar, Date = load_test(data_dir)
    points = pd.read_excel(r'.\data\resample_data.xlsx').columns[1:]
    test_forcast(model, x_test, covariates, params, scalar, Date, points)

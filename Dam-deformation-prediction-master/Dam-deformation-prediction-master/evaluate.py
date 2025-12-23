import numpy as np
import torch
import utils
import logging
from tqdm import tqdm
logger = logging.getLogger('MYAR.Eval')


def evaluate(model, loss_fn, val_loader, params, sample=True):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():
        raw_metrics = utils.init_metrics(sample=sample)

        for i, (val_input, val_label) in enumerate(tqdm(val_loader)):
            val_input = val_input.permute(1, 0, 2, 3).to(torch.float32).to(params.device)
            val_label = val_label.to(torch.float32).to(params.device)
            # label要去标准化
            batch_size = val_input.shape[1]

            # 存放在lag_days端的预测值，长度为lag_days-1
            input_mu = torch.zeros(batch_size, params.lag_days-1, params.num_series, device=params.device)
            input_sigma = torch.zeros(batch_size, params.lag_days-1, params.num_series, device=params.device)

            hidden = model.init_hidden(batch_size*params.num_series)
            cell = model.init_cell(batch_size*params.num_series)
            for t in range(params.lag_days-1):

                mu, sigma, hidden, cell = model(val_input[t].clone(), hidden, cell)
                # input_mu, input_sigma要去标准化
                input_mu[:, t, :] = mu.clone()
                input_sigma[:, t, :] = sigma.clone()

            if sample:
                # samples[sample_times, B, T, Nodes_num]   sample_mu[B, T, Nodes_num]
                samples, sample_mu, sample_sigma = model.test(val_input, hidden, cell, sampling=True)
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, val_label,
                                                   params.for_days, samples, relative=params.relative_metrics)

            else:
                sample_mu, sample_sigma = model.test(val_input, hidden, cell)
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, val_label,
                                                   params.for_days, relative=params.relative_metrics)

        summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
        metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
        logger.info('- Full val metrics: ' + metrics_string)
    return summary_metric

import argparse
import utils
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import my_model
from numpy import ndarray
import pickle
import time
from dataloader import *
from evaluate import evaluate
from torch.optim.lr_scheduler import LambdaLR


def Spit_gen(data_dir, ratio):
    x_input = np.load(os.path.join(data_dir, 'train_x_input.npy'))
    samples_num = x_input.shape[0]
    shuffled_indices = np.random.permutation(samples_num)
    # shuffled_indices = np.arange(samples_num)
    train_idx = shuffled_indices[:int(ratio * samples_num)]
    val_idx = shuffled_indices[int(ratio * samples_num):]
    return train_idx, val_idx


def Read_scalar(data_dir):
    scalar = np.load(os.path.join(data_dir, r'scalar.npy'))
    return scalar


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          params: utils.Params,
          epoch: int) -> ndarray:
    """
    Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    """
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    # Train_loader:
    # train_input: B, windows-1, Nodes_num, 1+cov_dim+1
    # train_label: B, windows-1, Nodes_num
    for i, (train_input, train_label) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_input.shape[0]
        loss = torch.zeros(1, device=params.device)

        train_input = train_input.permute(1, 0, 2, 3).to(torch.float32).to(params.device)
        train_label = train_label.permute(1, 0, 2).to(torch.float32).to(params.device)

        hidden = model.init_hidden(batch_size * params.num_series)
        cell = model.init_cell(batch_size * params.num_series)

        # mu[B, Nodes_num]  # sigma[B, Nodes_num]
        for t in range(params.window_days - 1):
            mu, sigma, hidden, cell = model(train_input[t].clone(), hidden, cell)
            loss += loss_fn(mu, sigma, train_label[t])  # 对第t步沿B求平均

        loss.backward()
        optimizer.step()
        loss = loss.item() / params.window_days
        loss_epoch[i] = loss
    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       params: utils.Params,
                       scalar: np.ndarray,
                       restore_file: str = None) -> None:
    '''Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        val_loader: load test data and labels
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    start_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    # fetch loss function
    loss_fn = my_model.loss_fn_gassion
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        start_epoch = utils.load_checkpoint(restore_path, model, optimizer)['epoch']
    logger.info('begin training and evaluation')
    best_val_ND = float('inf')
    train_len = len(train_loader)
    ND_summary = np.zeros(params.num_epochs)
    loss_summary = np.zeros((train_len * params.num_epochs))

    for epoch in np.arange(start_epoch, params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader, params,
                                                                        epoch)
        scheduler.step()

        val_metrics = evaluate(model, loss_fn, val_loader, params, sample=args.sampling)  # 验证时在forecast_days部分采用滚动预测
        ND_summary[epoch] = val_metrics['ND']

        is_best = ND_summary[epoch] <= best_val_ND  # bool 判断此轮是否训练到此的最优轮
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=is_best,
                              checkpoint=params.model_dir)

        if is_best:
            logger.info('- Found new best ND, 发现最优为第{}轮， 存储在epoch_{}.pth.tar内'.format(epoch, epoch))
            best_val_ND = ND_summary[epoch]
            best_json_path = os.path.join(params.model_dir, 'metrics_val_best_weights.json')
            utils.save_dict_to_json(val_metrics, best_json_path)

        logger.info('Current Best ND is: %.5f' % best_val_ND)

        utils.plot_all_epoch(ND_summary[:epoch + 1], 'ND', params.plot_dir)
        utils.plot_all_epoch(loss_summary[:(epoch + 1) * train_len], 'loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir, 'metrics_val_last_weights.json')
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    logger = logging.getLogger('MYAR.Train')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
    parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
    parser.add_argument('--restore-file', default=None, help='Optional, name of the file in --model_dir containing weights to reload before \
                        training')  # 'best' or 'epoch_#'
    parser.add_argument('--relative-metrics', default=False, help='Whether to normalize the metrics by label scales')
    parser.add_argument('--sampling', default=False, help='Whether to sample during evaluation')
    # Load the parameters from json file
    args = parser.parse_args()

    #  选择哪些图结构信息
    graph_choose = {'DTW': True,
                    'Gauss': True,
                    'Partition': True,
                    'Line': True}

    model_dir = os.path.join('experiments', args.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder)
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)
    params.graph_choose = graph_choose
    params.relative_metrics = args.relative_metrics
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')

    # 读取图矩阵信息 包括A_head和sqr_D_head
    with open(os.path.join(data_dir, 'Graph_info.pkl'), "rb") as tf:
        graph_infor = pickle.load(tf)  # 读取了所有的graph_infor信息，在模型中进行筛选

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


    # def get_parameter_number(model):
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(name, ':', param.size())
    #     total_num = sum(p.numel() for p in model.parameters())
    #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     num = {'Total': total_num, 'Trainable': trainable_num}
    #     print(num)
    # get_parameter_number(model)
    # exit(0)

    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logger.info('Loading the datasets...')

    train_index, val_index = Spit_gen(data_dir, params.split_ratio)
    train_set = Data_set(data_dir, train_index, True)
    val_set = Data_set(data_dir, val_index, False)
    scalar = Read_scalar(data_dir)

    # sampler = WeightedSampler(data_dir, train_index)  # Use weighted sampler instead of random sampler
    # train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=sampler, num_workers=4)
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=RandomSampler(train_set), num_workers=4)
    val_loader = DataLoader(val_set, batch_size=params.predict_batch, sampler=RandomSampler(val_set), num_workers=4)
    logger.info('Loading complete.')
    logger.info('Model: \n{}'.format(str(model)))

    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    # start_time = time.time()
    # train_time = []
    train_and_evaluate(model,
                       train_loader,
                       val_loader,
                       params,
                       scalar,
                       args.restore_file)
    # end_time = time.time()
    # train_time.append(end_time-start_time)
    # with open(model_dir+r'\time.txt', 'w') as q:
    #     for e in range(len(train_time)):
    #         t = str(train_time[e])
    #         q.write(t.strip(' '))
    #         q.write('\n')

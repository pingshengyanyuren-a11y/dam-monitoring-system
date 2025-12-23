import pandas as pd
import numpy as np
import os
import datetime
import utils
from tqdm import trange
import pickle
import copy


def gen_covariates_Date(date_start, date_train_last, date_end, for_days, resample_interval):
    # 生成训练期与预测期的时间戳
    last_day = copy.copy(date_train_last)
    while last_day < date_end:
        last_day += datetime.timedelta(days=(resample_interval * for_days))
    date = []
    date_ele = copy.copy(date_start)
    while date_ele <= last_day:
        date.append(date_ele)
        date_ele += datetime.timedelta(days=resample_interval)
    extra_num = (last_day - date_end).days // resample_interval

    return date, extra_num


def re_set_Day(date_start, data_train_last, date_end, resample_freq):
    # 根据重采样间隔，重新确定data_train_last, date_end，其中data_train_last，date_end要正好小于原来的data_train_last和date_en
    date1 = copy.copy(date_start)
    date2 = copy.copy(date_start)
    train_last, end = 0, 0
    while date1 <= data_train_last:
        train_last = date1
        date1 += datetime.timedelta(days=resample_freq)

    while date2 <= date_end:
        end = date2
        date2 += datetime.timedelta(days=resample_freq)
    return date_start, train_last, end


def point_data_deal(point_dir, points):
    df = pd.read_excel(point_dir, sheet_name='Sheet1')
    df = df.set_index('point').dropna(axis=1)
    df = df.reindex(points)

    filling_area = df['填筑区'].values
    filling_reference = df['positin_h'].values

    time_reference = df['时效量基准日期'].dt.date.values
    return {'filling_area': filling_area, 'filling_reference': filling_reference, 'time_reference': time_reference}


class Cov_resample():
    def __init__(self, data_dir, cov_name, date_start, date_train_last, date_end, resample_freq, for_days):
        self.data_dir = data_dir
        self.save_dir = r'.\resample_' + cov_name + '_data.xlsx'
        self.cov_name = cov_name
        self.for_days = for_days
        self.date_start = date_start
        self.date_end = date_end
        self.date_train_last = date_train_last
        self.resample_freq = resample_freq
        self.df_new, self.last_day = self.data_resample()
        self.resample_date = self.gen_Date(date_start)

    def gen_Date(self, date_start):
        date = []
        while date_start <= self.last_day:
            date.append(date_start)
            date_start += datetime.timedelta(days=self.resample_freq)
        return date

    def data_resample(self, ):
        last_days = copy.copy(self.date_train_last)
        df = pd.read_excel(self.data_dir, index_col='Date', sheet_name=self.cov_name)
        try:
            df = df.resample('D').mean()
        except:
            pass
        df = df.dropna(axis=0, how='all')
        df = df.reset_index(drop=False)
        cov_last_days = df['Date'].get(len(df['Date']) - 1)
        while last_days < self.date_end:
            last_days += datetime.timedelta(days=(self.resample_freq * self.for_days))
        if cov_last_days < last_days:
            exit()

        i = 0
        df_new = None
        for column in df.columns:
            if column == 'Date':
                continue
            if i == 0:
                df_new = self.data_Full(df['Date'], df[column], column)
            else:
                df_new = pd.merge(df_new, self.data_Full(df['Date'], df[column], column), left_on="Date",
                                  right_on="Date")
            i += 1
        return df_new, last_days

    def get_Date_resample(self):
        df_resample = pd.DataFrame()
        df = self.df_new
        df.index = range(len(df))  # 需对df重新设置索引 索引被重置也不知道什么原因 玄学
        Date_index = []
        for index, date in enumerate(df['Date'].tolist()):
            if date in self.resample_date:
                Date_index.append(index)
        for column in df.columns:
            df_resample[column] = df[column][Date_index]
        try:
            df_resample.to_excel(self.save_dir, index=None)
        except:
            pass
        return df_resample

    def data_Full(self, df1_date, df1_data, column):
        df1 = pd.DataFrame()
        df1['Date'] = df1_date
        df1[column] = df1_data
        df = df1.iloc[:, 0:2].dropna()
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        helper = pd.DataFrame(
            {'Date': pd.date_range(df['Date'].min(), df['Date'].max())})  # 生成辅助dataframe，包含所有日期
        df_int = pd.merge(df, helper, on='Date', how='outer').sort_values('Date')  # 融合两个DataFrame
        df_int.iloc[:, 1] = df_int.iloc[:, 1].interpolate(method='linear')  # 线性插值
        df_int.iloc[:, 1] = df_int.iloc[:, 1].round(4)
        return df_int

class Data_resample():
    def __init__(self, data_dir, save_dir, date_start, date_train_last, date_end, resample_freq, for_days):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.for_days = for_days
        self.date_start = date_start
        self.date_end = date_end
        self.date_train_last = date_train_last
        self.resample_freq = resample_freq
        self.df_new = self.data_resample()
        self.resample_date = self.gen_Date(date_start, date_end)

    def gen_Date(self, date_start, date_end):
        date = []
        #  需要根据所选择的日期间隔进行调整
        while date_start <= date_end:
            date.append(date_start)
            date_start += datetime.timedelta(days=self.resample_freq)
        return date

    def split_data(self, df):
        D_index = []
        for index, ele in enumerate(df.columns):
            if '日期' in ele:
                D_index.append(index)
        D_index.append(len(df.columns))
        split_df = []
        for i in range(len(D_index) - 1):
            df1 = df[df.columns[D_index[i]:D_index[i + 1]]].dropna(axis=0, 
                                                                   thresh=None, subset=None, inplace=False)
            df1 = df1.rename(columns={df1.columns[0]: 'Date'}, inplace=False)
            df1['Date'] = df1['Date'].dt.date
            split_df.append(df1)
        return split_df

    def fill_tail(self, dfs):
        df_new = []
        for df in dfs:
            if self.date_end > df.loc[len(df) - 1]["Date"]:
                fill = [self.date_end]
                for i in range(len(df.columns) - 1):
                    fill.append(0)
                df.loc[len(df)] = fill
            df_new.append(df)
        return df_new

    def data_resample(self, ):
        data_dir1 = r'.\data.xlsx'
        dfs = pd.read_excel(data_dir1)
        dfs = self.split_data(dfs)
        dfs = self.fill_tail(dfs)
        dfs_1_inter = []

        for df in dfs:
            i = 0
            df_new = None
            for column in df.columns:
                if column == 'Date':
                    continue
                if i == 0:
                    df_new = self.data_Full(df['Date'], df[column], column)  # 在此考虑对起始长短不一是否对起始top处进行fill
                else:
                    df_new = pd.merge(df_new, self.data_Full(df['Date'], df[column], column), left_on="Date",
                                      right_on="Date")
                i += 1
            dfs_1_inter.append(df_new)

        return dfs_1_inter

    def get_Date_resample(self):
        df_resample = pd.DataFrame()
        df_resample['Date'] = self.resample_date
        dfs_1_inter = self.df_new
        for df in dfs_1_inter:
            df.index = range(len(df))  # 需对df重新设置索引 索引被重置也不知道什么原因 玄学
            Date_index = []
            for index, date in enumerate(df['Date'].tolist()):
                if date in self.resample_date:
                    Date_index.append(index)
            for column in df.columns[1:]:
                df_resample[column] = df[column][Date_index].values
        try:
            df_resample.to_excel(self.save_dir, index=None)
        except:
            pass
        return df_resample

    # 缺失值处理，插值替换,频率为1天
    def data_Full(self, df1_date, df1_data, column):
        df1 = pd.DataFrame()
        df1['Date'] = df1_date
        df1[column] = df1_data
        df = df1.iloc[:, 0:2].dropna()
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        helper = pd.DataFrame(
            {'Date': pd.date_range(df['Date'].min(), df['Date'].max())})  # 生成辅助dataframe，包含所有日期
        df_int = pd.merge(df, helper, on='Date', how='outer').sort_values('Date')  # 融合两个DataFrame
        df_int.iloc[:, 1] = df_int.iloc[:, 1].interpolate(method='linear')  # 线性插值
        df_int.iloc[:, 1] = df_int.iloc[:, 1].round(4)
        return df_int


def gen_covariates(covariates_Date, filling, water, points_info):
    """
    :param covariates_Date:
    :param filling:
    :param points_info:
    :return:  covariates N*T*D, 归一化后
    """
    covariates = []
    points_num = len(points_info['filling_area'])
    for p in range(points_num):
        time_list = np.zeros(len(covariates_Date))
        for i in range(len(covariates_Date)):
            time_list[i] = ((covariates_Date[i] - points_info['time_reference'][p]).days + 1) / 100
        covariate_0 = (time_list - np.min(time_list)) / (np.max(time_list) - np.min(time_list))
        covariate_1 = np.log(time_list)  # log_total_time
        covariate_1 = (covariate_1 - np.min(covariate_1)) / (np.max(covariate_1) - np.min(covariate_1))


        water_l = np.array(water['上游水位'].values) - points_info['filling_reference'][p]
        water_l1 = (water_l - min(water_l)) / (max(water_l) - min(water_l))


        covariate = np.stack([covariate_0, covariate_1, water_l1], axis=1)
        covariates.append(covariate)
    covariates = np.stack(covariates, axis=0)
    return covariates


def prep_train_data(data, covariates):
    # covariates: N*T*D
    num_series, num_covariates = covariates.shape[0], covariates.shape[2]
    time_len = data.shape[0]
    all_windows = (time_len - windows_days + stride_days) // stride_days
    windows_index = np.arange(0, all_windows)

    # x_input的特征维度 0：自身  1：total_time  2：log_total_time  3：sin_time 4: cos_time 5: series_num
    # 为适应图数据, x_input：[all_samples, T, Nodes, Dim], T = windows_days - 1
    x_input = np.zeros((len(windows_index), windows_days - 1, num_series, 1 + num_covariates + 1), dtype='float32')
    # label windows内每一步的预测值[all_samples, T, Nodes]
    label = np.zeros((len(windows_index), windows_days - 1, num_series), dtype='float32')
    # v_input: [Nodes, 2], max, min
    scalar = np.zeros((num_series, 2), dtype='float32')

    for series in range(num_series):
        scalar[series] = [max(data[:, series]), min(data[:, series])]

    covariates = covariates[:, :time_len]
    for index in trange(len(windows_index)):
        for series in range(num_series):
            window_start = stride_days * index

            x_input[index, :, series, 0] = (data[window_start:window_start + windows_days - 1, series] - scalar[
                series, 1]) / (
                                                   scalar[series, 0] - scalar[series, 1])
            x_input[index, :, series, 1:1 + num_covariates] = covariates[series, window_start + 1:window_start + windows_days, :]
            x_input[index, :, series, -1] = series

            label[index, :, series] = (data[window_start + 1:window_start + windows_days, series] - scalar[
                series, 1]) / (scalar[series, 0] - scalar[series, 1])

    prefix = os.path.join('train_')
    np.save(prefix + 'x_input', x_input)
    np.save("scalar", scalar)
    np.save(prefix + 'label', label)


def prep_test_data(test_data, covariates, extra_num):
    x_test = test_data
    time_len = x_test.shape[0]
    # covariate的起始时间与用于预测的test_data在train_last_data部分的起始时间一致， 所以covariate第一个数据用不到
    covariate = covariates[:, -(time_len + extra_num):]
    prefix = os.path.join('test_')
    np.save(prefix + 'x', x_test)  # T, Nodes_nums 未归一化
    np.save(prefix + 'covariate', covariate)  # T, cov_nums


if __name__ == '__main__':
    data_dir = r'.\data.xlsx'
    data_save_dir = r'.\resample_data.xlsx'
    cov_dir = r'.\cov_data.xlsx'
    water_dir = r'.\water_level.xlsx'
    point_dir = r'.\point_data.xlsx'
    date_start = datetime.date(2022, 1, 1)  # 蓄水信息从2007.12.30日开始
    date_train_last = datetime.date(2024, 2, 25)
    date_end = datetime.date(2024, 6, 25)
    # 重新采样日期形式 (15, 30)为半月采，np.arange(2,31,2)为两日一采
    resample_interval = 5
    date_start, date_train_last, date_end = re_set_Day(date_start, date_train_last, date_end, resample_interval)

    model_dir = os.path.join('experiments', 'base_model')
    json_path = os.path.join(model_dir, 'params.json')
    params = utils.Params(json_path)

    num_covariates = params.cov_dim
    stride_days = params.stride_days
    lag_days = params.lag_days
    for_days = params.for_days
    windows_days = lag_days + for_days
    num_series = params.num_series

    data_resample = Data_resample(data_dir, data_save_dir, date_start, date_train_last, date_end, resample_interval, for_days).get_Date_resample()
    data_resample = data_resample.set_index('Date')
    filling = Cov_resample(cov_dir, 'filling', date_start, date_train_last, date_end, resample_interval, for_days).get_Date_resample()
    water = Cov_resample(water_dir, 'water', date_start, date_train_last, date_end, resample_interval, for_days).get_Date_resample()

    points = data_resample.columns
    points_info = point_data_deal(point_dir, points)

    covariates_Date, extra_num = gen_covariates_Date(date_start, date_train_last, date_end, for_days, resample_interval)
    covariates = gen_covariates(covariates_Date, filling, water, points_info)  # 不同测点的起始日期不一致，因此同一时刻对应的时效特征也有差别

    # 训练集需要进一步划分为测试集和训练集
    prep_train_data(data_resample[date_start:date_train_last].values, covariates)  # 对训练数据构造数据集

    train_last_data = data_resample[date_start:date_train_last].values[-lag_days:-1]
    test_data = data_resample[date_train_last:date_end].values
    test_Date = data_resample[date_train_last:date_end].index
    np.save('test_Date', test_Date)
    test_data = np.concatenate([train_last_data, test_data], 0)
    prep_test_data(test_data, covariates, extra_num)

import pandas as pd
import numpy as np
import datetime
import copy
from fastdtw import fastdtw
import pickle


class Data_resample():
    def __init__(self, data_df, date_start, date_end, resample_freq):
        self.data = data_df
        self.date_start = date_start
        self.date_end = date_end
        self.resample_freq = resample_freq
        self.df_new = self.data_resample()
        self.resample_date = self.gen_Date(date_start, date_end)  # 生成start至end期间的重采样日期

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
            df1 = df[df.columns[D_index[i]:D_index[i + 1]]].dropna(axis=0, how='all',
                                                                   thresh=None, subset=None, inplace=False)
            df1 = df1.rename(columns={df1.columns[0]: 'Date'}, inplace=False)
            df1['Date'] = df1['Date'].dt.date
            split_df.append(df1)
        return split_df

    def data_resample(self, ):
        dfs = self.data
        dfs = self.split_data(dfs)  # 按相同日期分离数据
        dfs_1_inter = []

        for df in dfs:
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
            dfs_1_inter.append(df_new)

        return dfs_1_inter

    def get_Date_resample(self):
        df_resample = pd.DataFrame()
        df_mask = pd.DataFrame()
        df_resample['Date'], df_mask['Date'] = self.resample_date, self.resample_date
        for df in self.df_new:
            df.index = range(len(df))  # 需对df重新设置索引 索引被重置也不知道什么原因 玄学
            Date_index = []
            Date = []
            for index, date in enumerate(df['Date'].dt.date.tolist()):
                if date in self.resample_date:
                    Date_index.append(index)
                    Date.append(date)
            pad_Date = list(set(self.resample_date) - set(Date))
            for column in df.columns[1:]:
                data_column = df[column][Date_index].values  # 因为计算距离需要对序列分别归一化，因此在该部分就先完成归一化工作
                data_column = (data_column-min(data_column))/(max(data_column)-min(data_column))
                column_resample = np.concatenate((np.array([np.NAN] * len(pad_Date)), data_column), axis=0)
                column_mask = np.bool_(np.concatenate((np.array([1] * len(pad_Date)), np.array([0] * len(Date))), axis=0))
                df_resample[column] = column_resample
                df_mask[column] = column_mask

        return df_resample, df_mask

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


def re_set_Day(date_start, data_last, resample_freq):
    # 根据重采样间隔，重新确定data_train_last, date_end，其中data_train_last，date_end要正好小于原来的data_train_last和date_en
    date1 = copy.copy(date_start)
    train_last = 0
    while date1 <= data_last:
        train_last = date1
        date1 += datetime.timedelta(days=resample_freq)

    return date_start, train_last


def get_data(time1, time2, resample_interval):
    time1, time2 = re_set_Day(time1, time2, resample_interval)
    data = pd.read_excel(r'.\data.xlsx')

    data_resample, df_mask = Data_resample(data, time1, time2, resample_interval).get_Date_resample()
    data_resample, df_mask = data_resample.set_index('Date'), df_mask.set_index('Date')
    return data_resample, df_mask


def DTW(s1, s2):
    euclidean_norm = lambda x, y: np.abs(x - y)
    distance, path = fastdtw(s1, s2, dist=euclidean_norm)
    return distance


def DTW_Adj_matrix(weight_threshold):
    time1 = datetime.date(2022, 1, 1)
    time2 = datetime.date(2024, 2, 25)
    resample_interval = 5
    data_resample, df_mask = get_data(time1, time2, resample_interval)  # 重采样后的序列df和填充掩码df
    points = data_resample.columns
    A = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(len(points)):
            s_i, s_j = data_resample[points[i]].values, data_resample[points[j]].values
            s_i, s_j = s_i[np.bool_(1-df_mask[points[i]])], s_j[np.bool_(1-df_mask[points[j]])]
            A[i, j] = DTW(s_i, s_j)
    A_head = 1/A
    A_head[A_head == np.inf] = 1  # 加权邻接矩阵
    A_head = np.where(A_head >= weight_threshold, A_head, 0)  # 大于阈值的权重才视为存在节点连接
    sqr_D_head = np.diag(1 / np.sqrt(np.sum(A_head, 1)))  # D^(-1/2)矩阵
    graph_infor = {'sqr_D_head': sqr_D_head,
                   'A_head': A_head}
    return graph_infor, points


def Gauss_Adj_matrix(points, weight_threshold=0.1):
    points_data = pd.read_excel(r'.\point_data.xlsx')
    points_data = points_data.set_index('point').dropna(axis=1)
    points_data = points_data.reindex(points)
    location = points_data[['positin_h', 'positin_ba', 'positin_zong']].values

    distcance_matrix = np.zeros((location.shape[0], location.shape[0]))
    for i in range(location.shape[0]):
        for j in range(location.shape[0]):
            distcance_matrix[i, j] = np.linalg.norm((location[i]-location[j]), ord=2)

    non_zero = distcance_matrix[distcance_matrix != 0]
    dis_std = np.std(non_zero)
    A_head = np.exp(-np.square(distcance_matrix) / np.square(dis_std))
    A_head = np.where(A_head >= weight_threshold, A_head, 0)  # 大于阈值的权重才视为存在节点连接
    sqr_D_head = np.diag(1 / np.sqrt(np.sum(A_head, 1)))
    graph_infor = {'sqr_D_head': sqr_D_head,
                   'A_head': A_head}
    return graph_infor


def Parti_Adj_matrix(points):
    points_data = pd.read_excel(r'.\point_data.xlsx')
    points_data = points_data.set_index('point').dropna(axis=1)
    points_data = points_data.reindex(points)
    partition = points_data['分区'].values
    A_head = np.zeros((partition.shape[0], partition.shape[0]))
    for i in range(partition.shape[0]):
        for j in range(partition.shape[0]):
            if partition[i] == partition[j]:
                A_head[i, j] = 1
    sqr_D_head = np.diag(1 / np.sqrt(np.sum(A_head, 1)))
    graph_infor = {'sqr_D_head': sqr_D_head,
                   'A_head': A_head}
    return graph_infor


def Line_Adj_matrix(points):
    points_data = pd.read_excel(r'.\point_data.xlsx')
    points_data = points_data.set_index('point').dropna(axis=1)
    points_data = points_data.reindex(points)
    partition = points_data['line'].values
    A_head = np.zeros((partition.shape[0], partition.shape[0]))
    for i in range(partition.shape[0]):
        for j in range(partition.shape[0]):
            if partition[i] == partition[j]:
                A_head[i, j] = 1
    sqr_D_head = np.diag(1 / np.sqrt(np.sum(A_head, 1)))
    graph_infor = {'sqr_D_head': sqr_D_head,
                   'A_head': A_head}
    return graph_infor


if __name__ == '__main__':
    DTW_threshold = 0.3
    DTW_Adj_matrix, points = DTW_Adj_matrix(DTW_threshold)

    Gauss_threshold = 0.5
    Gauss_Adj_matrix = Gauss_Adj_matrix(points, Gauss_threshold)

    Parti_Adj_matrix = Parti_Adj_matrix(points)
    Line_Adj_matrix = Line_Adj_matrix(points)
    graph_infor = {'DTW': DTW_Adj_matrix,
                   'Gauss': Gauss_Adj_matrix,
                   'Partition': Parti_Adj_matrix,
                   'Line': Line_Adj_matrix}
    with open("Graph_info.pkl", "wb") as tf:
        pickle.dump(graph_infor, tf)



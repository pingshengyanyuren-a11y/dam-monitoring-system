import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class FeatureEngineer:
    def __init__(self, df):
        """
        初始化特征工程类
        :param df: 包含 [Time_Step, Node_ID, X, Y, Total_Settlement] 等列的 DataFrame
        """
        self.df = df.copy()

    def add_temporal_features(self):
        """
        添加时序特征: Lag-1, Lag-2, Rolling Mean (window=3)
        基于 'Total_Settlement' 列。
        """
        print("正在计算时序特征 (Lag & Rolling)...")
        # 确保按 Node_ID 和 Time_Step 排序
        self.df.sort_values(by=['Node_ID', 'Time_Step'], inplace=True)
        
        # 定义目标列
        target_col = 'Total_Settlement'
        
        # 使用 groupby 进行操作
        grouped = self.df.groupby('Node_ID')[target_col]
        
        # Lag features
        self.df['lag_1'] = grouped.shift(1)
        self.df['lag_2'] = grouped.shift(2)
        
        # Rolling mean (window=3, min_periods=3 to avoid partial windows if strict)
        # Shift 1 to ensure we are using PAST data for prediction if that's the goal.
        # User request: "calculate rolling_mean_3 for past 3 steps".
        # Typically rolling mean includes current row by default in pandas.
        # If the goal is "feature for predicting current t", it should be shift(1).rolling(3).
        # However, usually rolling features are just added and user decides leakage.
        # Assuming typical setup: rolling window includes current row, but purely historical requires shift.
        # To be safe for prediction tasks: rolling mean of *previous* values.
        # But if the user just asks "rolling stats", standard is rolling on the series.
        # Let's assume standard rolling on the column, but caveat: if used for training t, it includes t.
        # "Past 3 time steps" implies t-1, t-2, t-3? Or t, t-1, t-2?
        # Let's implement rolling(3) on the series. If it needs to be strictly past, we can shift.
        # Given "lag_1", "lag_2", let's make rolling_mean_3 be the mean of (t, t-1, t-2) or (t-1, t-2, t-3).
        # Let's use standard rolling(3) which ends at t.
        self.df['rolling_mean_3'] = grouped.rolling(window=3).mean().reset_index(0, drop=True)
        
        print("时序特征计算完成。")

    def add_spatial_features(self):
        """
        添加空间邻域特征 (Spatial KNN)
        对每个时间步，计算每个节点最近 5 个邻居的平均变形量。
        """
        print("正在计算空间邻域特征 (Spatial KNN)...")
        
        # 结果容器
        self.df['neighbor_avg_disp'] = 0.0
        
        # 获取所有时间步
        time_steps = self.df['Time_Step'].unique()
        
        for t in time_steps:
            # 切片当前时间步
            mask = self.df['Time_Step'] == t
            step_df = self.df[mask].copy()
            
            # 提取坐标和目标值
            coords = step_df[['X', 'Y']].values
            values = step_df['Total_Settlement'].values
            
            if len(step_df) < 6:
                # 样本过少无法计算 5 个邻居
                continue
                
            # KNN (k=6, 因为包含自身)
            knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
            knn.fit(coords)
            
            # 找到最近邻索引
            distances, indices = knn.kneighbors(coords)
            
            # 计算邻居平均值 (排除自身，即 indices[:, 1:])
            # indices shape: (n_samples, 6)
            neighbor_indices = indices[:, 1:]
            
            # 获取邻居的值
            neighbor_values = values[neighbor_indices] # shape: (n_samples, 5)
            
            # 计算平均
            avg_disp = np.mean(neighbor_values, axis=1)
            
            # 填回原 DataFrame
            # 注意: 直接赋值需要索引对齐
            self.df.loc[mask, 'neighbor_avg_disp'] = avg_disp
            
        print("空间特征计算完成。")

    def process(self):
        """执行所有特征工程步骤并清洗数据"""
        self.add_temporal_features()
        self.add_spatial_features()
        
        # 处理 NaN
        # Lag-2 和 Rolling-3 会导致前两行即使有数据也可能 NaN (shift(2) -> 前两个NaN)
        # Drop rows with NaN created by lag/rolling
        initial_len = len(self.df)
        self.df.dropna(subset=['lag_1', 'lag_2', 'rolling_mean_3'], inplace=True)
        dropped_len = initial_len - len(self.df)
        
        print(f"数据清洗完成。移除了 {dropped_len} 行 (由于时序特征缺失)。")
        
        return self.df

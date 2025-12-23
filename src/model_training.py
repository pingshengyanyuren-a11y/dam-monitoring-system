import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 动态获取路径
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
data_path = os.path.join(project_root, "data", "processed", "training_dataset.csv")

def load_and_split_data(path):
    print(f"正在加载数据: {path}")
    df = pd.read_csv(path)
    
    # 确保按时间排序
    df.sort_values(by=['Time_Step', 'Node_ID'], inplace=True)
    
    # 定义特征和目标
    # 特征: X, Y, Time_Step, lag_1, lag_2, rolling_mean_3, neighbor_avg_disp
    # 注意: lag_1, lag_2, rolling_mean_3, neighbor_avg_disp 是之前特征工程生成的列名
    feature_cols = ['X', 'Y', 'Time_Step', 'lag_1', 'lag_2', 'rolling_mean_3', 'neighbor_avg_disp']
    target_col = 'Total_Settlement'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 按时间划分 (前80%训练，后20%测试)
    # 不使用 train_test_split 的 shuffle，而是基于时间步切分
    time_steps = df['Time_Step'].unique()
    split_idx = int(len(time_steps) * 0.8)
    split_time = time_steps[split_idx]
    
    print(f"划分时间点: T={split_time} (包含T之后的为测试集)")
    
    train_mask = df['Time_Step'] < split_time
    test_mask = df['Time_Step'] >= split_time
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def build_stacking_model():
    print("构建 Stacking 模型...")
    
    # Base Learners
    estimators = [
        ('lgbm', LGBMRegressor(n_estimators=1000, random_state=42, verbose=-1)),
        ('xgb', XGBRegressor(tree_method='hist', n_estimators=1000, random_state=42, verbosity=0)),
        ('cat', CatBoostRegressor(verbose=0, random_state=42))
    ]
    
    # Meta Learner
    # cv=5 by default for RidgeCV
    final_estimator = RidgeCV()
    
    # Stacking
    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        n_jobs=-1, # 并行计算
        passthrough=False # Meta learner 只看 base learner 的输出
    )
    
    return reg

def evaluate_model(model, X_test, y_test):
    print("正在评估模型...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 转换为毫米 (mm) 进行对比
    rmse_mm = rmse * 1000
    r2_mm = r2_score(y_test * 1000, y_pred * 1000)
    
    print("\n" + "="*40)
    print(f"模型评估结果 (Stacking):")
    print(f"[米 m]   RMSE: {rmse:.8f} | R2: {r2:.8f}")
    print(f"[毫米 mm] RMSE: {rmse_mm:.8f} | R2: {r2_mm:.8f}")
    print("="*40 + "\n")
    
    return rmse, r2

def predict_stacking(model, X_new):
    """封装的预测函数"""
    return model.predict(X_new)

def main():
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        return

    # 1. 准备数据
    X_train, y_train, X_test, y_test = load_and_split_data(data_path)
    
    # 2. 构建模型
    model = build_stacking_model()
    
    # 3. 训练
    print("开始训练 (这可能需要几分钟)...")
    model.fit(X_train, y_train)
    print("训练完成。")
    
    # 4. 评估
    rmse, r2 = evaluate_model(model, X_test, y_test)
    
    # 验证目标
    if r2 > 0.99:
        print("✅ 成功达成目标: R2 > 0.99")
    else:
        print("⚠️ 未达成目标: R2 <= 0.99，可能需要优化。")

if __name__ == "__main__":
    main()

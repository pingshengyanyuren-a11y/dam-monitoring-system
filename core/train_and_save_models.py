"""
训练脚本：基于 (X, Y, Time) 的土石坝变形预测模型 - 双目标版本
输入特征：X坐标、Y坐标、时间步、沉降滞后项、水平位移滞后项
输出：
1. 预测沉降量 (Total_Settlement)
2. 预测顺河向位移 (Cum_Disp_X)

不使用水位、温度数据。
"""

import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# Stacking 相关
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "processed", "master_dataset.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

# 确保目录存在
os.makedirs(MODELS_DIR, exist_ok=True)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================
# 1. 数据加载与特征工程
# ==============================
def load_and_prepare_data():
    print(f"加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"原始数据形状: {df.shape}")
    
    # 按节点和时间排序
    df.sort_values(by=['Node_ID', 'Time_Step'], inplace=True)
    
    # ------------------------------------
    # (1) 沉降特征 (Settlement Lags)
    # ------------------------------------
    for lag in [1, 2, 3, 5]:
        df[f'Lag_Settlement_{lag}'] = df.groupby('Node_ID')['Total_Settlement'].shift(lag)
    
    df['Rolling_Mean_Settlement_5'] = df.groupby('Node_ID')['Total_Settlement'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # ------------------------------------
    # (2) 水平位移特征 (Horizontal Lags)
    # ------------------------------------
    for lag in [1, 2, 3, 5]:
        df[f'Lag_Horizontal_{lag}'] = df.groupby('Node_ID')['Cum_Disp_X'].shift(lag)
        
    df['Rolling_Mean_Horizontal_5'] = df.groupby('Node_ID')['Cum_Disp_X'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # 删除 NaN
    df.dropna(inplace=True)
    print(f"特征工程后数据形状: {df.shape}")
    
    return df


# ==============================
# 2. Stacking 模型训练 (单目标)
# ==============================
def train_stacking_model(X_train, y_train, target_name="Target"):
    print(f"\n========== 训练 Stacking 模型 ({target_name}) ==========")
    
    # 定义基学习器
    base_learners = [
        ('lgbm', lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, verbosity=0, random_state=42)),
        ('catboost', CatBoostRegressor(n_estimators=100, verbose=0, random_state=42))
    ]
    
    # 元学习器
    meta_learner = Ridge()
    
    # Stacking
    stack_model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    print("开始拟合...")
    stack_model.fit(X_train, y_train)
    print(f"Stacking 模型 ({target_name}) 训练完成。")
    
    return stack_model


# ==============================
# 3. BiLSTM 模型定义 (双输出)
# ==============================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        att_weights = torch.softmax(self.u(u), dim=1)
        context = torch.sum(att_weights * x, dim=1)
        return context, att_weights


class AttentionBiLSTM_Dual(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AttentionBiLSTM_Dual, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        
        # 共享层
        self.fc_shared = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.2)
        
        # 输出层1: 沉降
        self.fc_settlement = nn.Linear(64, 1)
        # 输出层2: 水平位移
        self.fc_horizontal = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, att_weights = self.attention(lstm_out)
        
        shared_out = torch.relu(self.fc_shared(context))
        shared_out = self.dropout(shared_out)
        
        out_settlement = self.fc_settlement(shared_out)
        out_horizontal = self.fc_horizontal(shared_out)
        
        return out_settlement, out_horizontal, att_weights


def create_sequences(X, y, window_size=5):
    """为 BiLSTM 创建时间序列样本 (y可以是多维)"""
    seqs, labels = [], []
    for i in range(len(X) - window_size):
        seqs.append(X[i:i+window_size])
        labels.append(y[i+window_size])
    return np.array(seqs), np.array(labels)


def train_bilstm_model_dual(X_train_seq, y_train_seq, input_dim, epochs=50):
    print("\n========== 训练 Dual-Output BiLSTM 模型 ==========")
    
    # 转换为 Tensor
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq), 
        torch.FloatTensor(y_train_seq) # shape: (batch, 2)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    model = AttentionBiLSTM_Dual(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # y_batch shape: (batch, 2) -> split to settlement, horizontal
            y_settlement_true = y_batch[:, 0].unsqueeze(1)
            y_horizontal_true = y_batch[:, 1].unsqueeze(1)
            
            optimizer.zero_grad()
            out_s, out_h, _ = model(X_batch)
            
            # 联合 Loss
            loss_s = criterion(out_s, y_settlement_true)
            loss_h = criterion(out_h, y_horizontal_true)
            loss = loss_s + loss_h
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    print("Dual-Output BiLSTM 模型训练完成。")
    return model


# ==============================
# 4. 主流程
# ==============================
def main():
    # 1. 加载数据
    df = load_and_prepare_data()
    
    # 定义特征列
    feature_cols = ['X', 'Y', 'Time_Step'] + \
                   [f'Lag_Settlement_{i}' for i in [1, 2, 3, 5]] + \
                   ['Rolling_Mean_Settlement_5'] + \
                   [f'Lag_Horizontal_{i}' for i in [1, 2, 3, 5]] + \
                   ['Rolling_Mean_Horizontal_5']
    
    print(f"特征数量: {len(feature_cols)}")
    
    # 定义目标列
    target_s = 'Total_Settlement'
    target_h = 'Cum_Disp_X'
    
    X = df[feature_cols].values
    y_s = df[target_s].values
    y_h = df[target_h].values
    
    # 2. 归一化
    scaler_X = MinMaxScaler()
    scaler_y_s = MinMaxScaler() # 沉降归一化
    scaler_y_h = MinMaxScaler() # 水平位移归一化
    
    X_scaled = scaler_X.fit_transform(X)
    y_s_scaled = scaler_y_s.fit_transform(y_s.reshape(-1, 1)).flatten()
    y_h_scaled = scaler_y_h.fit_transform(y_h.reshape(-1, 1)).flatten()
    
    # 组合双目标 (用于BiLSTM)
    y_dual_scaled = np.column_stack((y_s_scaled, y_h_scaled))
    
    # 3. 划分数据集 (按时间顺序 80/20)
    split_idx = int(len(X_scaled) * 0.8)
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_s_train, y_s_test = y_s_scaled[:split_idx], y_s_scaled[split_idx:]
    y_h_train, y_h_test = y_h_scaled[:split_idx], y_h_scaled[split_idx:]
    y_dual_train, y_dual_test = y_dual_scaled[:split_idx], y_dual_scaled[split_idx:]
    
    print(f"\n训练集样本: {len(X_train)}, 测试集样本: {len(X_test)}")
    
    # ----------------------------------------
    # 4. 训练 Stacking 模型 (分别训练两个)
    # ----------------------------------------
    
    # (A) Stacking - 沉降
    stack_model_s = train_stacking_model(X_train, y_s_train, "沉降")
    
    # (B) Stacking - 水平位移
    stack_model_h = train_stacking_model(X_train, y_h_train, "水平位移")
    
    # 评估 Stacking
    print("\n--- Stacking 评估 ---")
    
    # 沉降
    pred_s = stack_model_s.predict(X_test)
    pred_s_real = scaler_y_s.inverse_transform(pred_s.reshape(-1, 1)).flatten()
    true_s_real = scaler_y_s.inverse_transform(y_s_test.reshape(-1, 1)).flatten()
    rmse_s = np.sqrt(mean_squared_error(true_s_real, pred_s_real))
    print(f"沉降预测 RMSE: {rmse_s*1000:.2f} mm")
    
    # 水平
    pred_h = stack_model_h.predict(X_test)
    pred_h_real = scaler_y_h.inverse_transform(pred_h.reshape(-1, 1)).flatten()
    true_h_real = scaler_y_h.inverse_transform(y_h_test.reshape(-1, 1)).flatten()
    rmse_h = np.sqrt(mean_squared_error(true_h_real, pred_h_real))
    print(f"水平预测 RMSE: {rmse_h*1000:.2f} mm")
    
    
    # ----------------------------------------
    # 5. 训练 Dual-Output BiLSTM 模型
    # ----------------------------------------
    window_size = 5
    X_train_seq, y_train_seq = create_sequences(X_train, y_dual_train, window_size)
    X_test_seq, y_test_seq = create_sequences(X_test, y_dual_test, window_size)
    
    bilstm_model = train_bilstm_model_dual(X_train_seq, y_train_seq, input_dim=len(feature_cols), epochs=50)
    
    # 评估 BiLSTM
    bilstm_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
        out_s, out_h, _ = bilstm_model(X_test_tensor)
        pred_lstm_s = out_s.cpu().numpy().flatten()
        pred_lstm_h = out_h.cpu().numpy().flatten()
        
    pred_lstm_s_real = scaler_y_s.inverse_transform(pred_lstm_s.reshape(-1, 1)).flatten()
    true_lstm_s_real = scaler_y_s.inverse_transform(y_test_seq[:, 0].reshape(-1, 1)).flatten()
    
    pred_lstm_h_real = scaler_y_h.inverse_transform(pred_lstm_h.reshape(-1, 1)).flatten()
    true_lstm_h_real = scaler_y_h.inverse_transform(y_test_seq[:, 1].reshape(-1, 1)).flatten()
    
    print("\n--- BiLSTM 评估 ---")
    print(f"沉降预测 RMSE: {np.sqrt(mean_squared_error(true_lstm_s_real, pred_lstm_s_real))*1000:.2f} mm")
    print(f"水平预测 RMSE: {np.sqrt(mean_squared_error(true_lstm_h_real, pred_lstm_h_real))*1000:.2f} mm")
    
    
    # ----------------------------------------
    # 6. 保存所有模型
    # ----------------------------------------
    print("\n保存模型...")
    
    # 保存 Stacking (沉降)
    with open(os.path.join(MODELS_DIR, "stacking_settlement.pkl"), 'wb') as f:
        pickle.dump({
            'model': stack_model_s,
            'scaler_X': scaler_X, 
            'scaler_y': scaler_y_s, 
            'features': feature_cols
        }, f)
        
    # 保存 Stacking (水平)
    with open(os.path.join(MODELS_DIR, "stacking_horizontal.pkl"), 'wb') as f:
        pickle.dump({
            'model': stack_model_h,
            # X scaler is same, but save it anyway for independence
            'scaler_X': scaler_X,
            'scaler_y': scaler_y_h,
            'features': feature_cols
        }, f)
        
    # 保存 BiLSTM
    torch.save({
        'model_state_dict': bilstm_model.state_dict(),
        'input_dim': len(feature_cols),
        'window_size': window_size,
        'scaler_X': scaler_X,      # 保存进去方便加载
        'scaler_y_s': scaler_y_s,
        'scaler_y_h': scaler_y_h
    }, os.path.join(MODELS_DIR, "bilstm_dual_model.pth"))
    
    print(f"所有模型已保存到: {MODELS_DIR}")


if __name__ == "__main__":
    main()

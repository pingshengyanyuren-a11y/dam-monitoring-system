"""
训练脚本 V2：修复代码与论文一致性问题
- 时序划分（前80%/后20%，无shuffle）
- 动态权重计算（基于RMSE倒数）
- 添加对比模型（MLR, SVR, 单独LSTM）
- 保存所有模型RMSE用于权重计算
"""

import pandas as pd
import numpy as np
import os
import pickle
import sqlite3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "processed", "master_dataset.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
BASELINE_DB_PATH = os.path.join(CURRENT_DIR, "data", "processed", "baseline_predictions.db")

os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================
# 1. 数据加载与特征工程
# ==============================
def load_and_prepare_data():
    print(f"加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"原始数据形状: {df.shape}")
    
    df.sort_values(by=['Node_ID', 'Time_Step'], inplace=True)
    
    # 沉降滞后特征
    for lag in [1, 2, 3, 5]:
        df[f'Lag_Settlement_{lag}'] = df.groupby('Node_ID')['Total_Settlement'].shift(lag)
    df['Rolling_Mean_Settlement_5'] = df.groupby('Node_ID')['Total_Settlement'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # 水平位移滞后特征
    for lag in [1, 2, 3, 5]:
        df[f'Lag_Horizontal_{lag}'] = df.groupby('Node_ID')['Cum_Disp_X'].shift(lag)
    df['Rolling_Mean_Horizontal_5'] = df.groupby('Node_ID')['Cum_Disp_X'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    df.dropna(inplace=True)
    print(f"特征工程后数据形状: {df.shape}")
    return df


# ==============================
# 2. 时序划分（前80%训练，后20%测试）
# ==============================
def temporal_train_test_split(df, test_ratio=0.2):
    """按时间顺序划分，不shuffle"""
    df_sorted = df.sort_values('Time_Step')
    split_idx = int(len(df_sorted) * (1 - test_ratio))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    print(f"时序划分: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条")
    return train_df, test_df


# ==============================
# 3. 对比模型训练
# ==============================
def train_baseline_models(X_train, y_train, X_test, y_test, target_name="Settlement"):
    """训练MLR, SVR对比模型并评估"""
    results = {}
    
    # 3.1 多元线性回归 (MLR)
    print(f"\n训练 MLR ({target_name})...")
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    pred_mlr = mlr.predict(X_test)
    rmse_mlr = np.sqrt(mean_squared_error(y_test, pred_mlr))
    r2_mlr = r2_score(y_test, pred_mlr)
    mae_mlr = mean_absolute_error(y_test, pred_mlr)
    results['MLR'] = {'model': mlr, 'rmse': rmse_mlr, 'r2': r2_mlr, 'mae': mae_mlr, 'pred': pred_mlr}
    print(f"  MLR: RMSE={rmse_mlr*1000:.2f}mm, R²={r2_mlr:.4f}")
    
    # 3.2 支持向量回归 (SVR)
    print(f"训练 SVR ({target_name})...")
    svr = SVR(kernel='rbf', C=100, gamma='scale')
    svr.fit(X_train, y_train)
    pred_svr = svr.predict(X_test)
    rmse_svr = np.sqrt(mean_squared_error(y_test, pred_svr))
    r2_svr = r2_score(y_test, pred_svr)
    mae_svr = mean_absolute_error(y_test, pred_svr)
    results['SVR'] = {'model': svr, 'rmse': rmse_svr, 'r2': r2_svr, 'mae': mae_svr, 'pred': pred_svr}
    print(f"  SVR: RMSE={rmse_svr*1000:.2f}mm, R²={r2_svr:.4f}")
    
    return results


# ==============================
# 4. Stacking模型训练
# ==============================
def train_stacking_model(X_train, y_train, target_name="Target"):
    print(f"\n========== 训练 Stacking 模型 ({target_name}) ==========")
    base_learners = [
        ('lgbm', lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, verbosity=0, random_state=42)),
        ('catboost', CatBoostRegressor(n_estimators=100, verbose=0, random_state=42))
    ]
    meta_learner = Ridge()
    stack_model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5, n_jobs=-1
    )
    stack_model.fit(X_train, y_train)
    print(f"Stacking 模型训练完成 ({target_name})")
    return stack_model


# ==============================
# 5. BiLSTM模型（带Attention）
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
        self.fc_shared = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc_settlement = nn.Linear(64, 1)
        self.fc_horizontal = nn.Linear(64, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, att_weights = self.attention(lstm_out)
        shared_out = torch.relu(self.fc_shared(context))
        shared_out = self.dropout(shared_out)
        out_s = self.fc_settlement(shared_out)
        out_h = self.fc_horizontal(shared_out)
        return out_s, out_h, att_weights

class SimpleLSTM(nn.Module):
    """单独LSTM用于对比实验"""
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


def create_sequences(X, y, window_size=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)


def train_lstm_single(X_train_seq, y_train_seq, input_dim, epochs=50):
    """训练单独LSTM用于对比"""
    model = SimpleLSTM(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(
        torch.FloatTensor(X_train_seq).to(device),
        torch.FloatTensor(y_train_seq).to(device)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"  单独LSTM Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    return model


def train_bilstm_dual(X_train_seq, y_train_seq, input_dim, epochs=50):
    """训练双输出BiLSTM"""
    model = AttentionBiLSTM_Dual(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    y_s = y_train_seq[:, 0]
    y_h = y_train_seq[:, 1]
    
    dataset = TensorDataset(
        torch.FloatTensor(X_train_seq).to(device),
        torch.FloatTensor(y_s).to(device),
        torch.FloatTensor(y_h).to(device)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, ys_batch, yh_batch in loader:
            optimizer.zero_grad()
            pred_s, pred_h, _ = model(X_batch)
            loss = criterion(pred_s.squeeze(), ys_batch) + criterion(pred_h.squeeze(), yh_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"  BiLSTM Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    return model


# ==============================
# 6. 创建对比结果数据库
# ==============================
def create_baseline_database(test_df, predictions, metrics):
    """创建baseline_predictions.db存储对比结果"""
    conn = sqlite3.connect(BASELINE_DB_PATH)
    cursor = conn.cursor()
    
    # 创建性能表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            model_name TEXT PRIMARY KEY,
            target TEXT,
            rmse REAL,
            r2 REAL,
            mae REAL
        )
    ''')
    
    # 创建预测表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS baseline_predictions (
            node_id INTEGER,
            time_step INTEGER,
            actual_settlement REAL,
            pred_mlr REAL,
            pred_svr REAL,
            pred_lstm REAL,
            pred_stacking REAL,
            pred_bilstm REAL,
            pred_hybrid REAL,
            PRIMARY KEY (node_id, time_step)
        )
    ''')
    
    # 保存性能指标
    for model_name, data in metrics.items():
        cursor.execute('''
            INSERT OR REPLACE INTO model_performance (model_name, target, rmse, r2, mae)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_name, data.get('target', 'Settlement'), 
              data['rmse']*1000, data['r2'], data['mae']*1000))
    
    # 保存预测结果
    for i, row in test_df.reset_index(drop=True).iterrows():
        if i < len(predictions.get('MLR', [])):
            cursor.execute('''
                INSERT OR REPLACE INTO baseline_predictions 
                (node_id, time_step, actual_settlement, pred_mlr, pred_svr, pred_lstm, 
                 pred_stacking, pred_bilstm, pred_hybrid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(row['Node_ID']), int(row['Time_Step']), 
                float(row['Total_Settlement']),
                float(predictions.get('MLR', [0]*len(test_df))[i]) if i < len(predictions.get('MLR',[])) else 0,
                float(predictions.get('SVR', [0]*len(test_df))[i]) if i < len(predictions.get('SVR',[])) else 0,
                float(predictions.get('LSTM', [0]*len(test_df))[i]) if i < len(predictions.get('LSTM',[])) else 0,
                float(predictions.get('Stacking', [0]*len(test_df))[i]) if i < len(predictions.get('Stacking',[])) else 0,
                float(predictions.get('BiLSTM', [0]*len(test_df))[i]) if i < len(predictions.get('BiLSTM',[])) else 0,
                float(predictions.get('Hybrid', [0]*len(test_df))[i]) if i < len(predictions.get('Hybrid',[])) else 0
            ))
    
    conn.commit()
    conn.close()
    print(f"\n对比结果数据库已保存: {BASELINE_DB_PATH}")


# ==============================
# 主函数
# ==============================
def main():
    # 1. 加载数据
    df = load_and_prepare_data()
    
    # 2. 时序划分（关键修复！）
    train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)
    
    # 特征列
    feature_cols = ['X', 'Y', 'Time_Step'] + \
                   [f'Lag_Settlement_{i}' for i in [1, 2, 3, 5]] + \
                   ['Rolling_Mean_Settlement_5'] + \
                   [f'Lag_Horizontal_{i}' for i in [1, 2, 3, 5]] + \
                   ['Rolling_Mean_Horizontal_5']
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train_s = train_df['Total_Settlement'].values
    y_test_s = test_df['Total_Settlement'].values
    y_train_h = train_df['Cum_Disp_X'].values
    y_test_h = test_df['Cum_Disp_X'].values
    
    # 归一化
    scaler_X = MinMaxScaler()
    scaler_y_s = MinMaxScaler()
    scaler_y_h = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_s_scaled = scaler_y_s.fit_transform(y_train_s.reshape(-1,1)).flatten()
    y_test_s_scaled = scaler_y_s.transform(y_test_s.reshape(-1,1)).flatten()
    y_train_h_scaled = scaler_y_h.fit_transform(y_train_h.reshape(-1,1)).flatten()
    y_test_h_scaled = scaler_y_h.transform(y_test_h.reshape(-1,1)).flatten()
    
    all_metrics = {}
    all_predictions = {}
    
    # ========================================
    # 3. 训练对比模型（MLR, SVR）
    # ========================================
    baseline_results = train_baseline_models(X_train_scaled, y_train_s_scaled, 
                                             X_test_scaled, y_test_s_scaled, "Settlement")
    
    for name, data in baseline_results.items():
        pred_real = scaler_y_s.inverse_transform(data['pred'].reshape(-1,1)).flatten()
        true_real = scaler_y_s.inverse_transform(y_test_s_scaled.reshape(-1,1)).flatten()
        rmse_real = np.sqrt(mean_squared_error(true_real, pred_real))
        r2_real = r2_score(true_real, pred_real)
        mae_real = mean_absolute_error(true_real, pred_real)
        all_metrics[name] = {'rmse': rmse_real, 'r2': r2_real, 'mae': mae_real, 'target': 'Settlement'}
        all_predictions[name] = pred_real
        print(f"【{name}】真实尺度: RMSE={rmse_real*1000:.2f}mm, R²={r2_real:.4f}")
    
    # ========================================
    # 4. 训练Stacking模型
    # ========================================
    stack_model_s = train_stacking_model(X_train_scaled, y_train_s_scaled, "Settlement")
    pred_stack_s = stack_model_s.predict(X_test_scaled)
    pred_stack_s_real = scaler_y_s.inverse_transform(pred_stack_s.reshape(-1,1)).flatten()
    rmse_stack_s = np.sqrt(mean_squared_error(y_test_s, pred_stack_s_real))
    r2_stack_s = r2_score(y_test_s, pred_stack_s_real)
    mae_stack_s = mean_absolute_error(y_test_s, pred_stack_s_real)
    all_metrics['Stacking'] = {'rmse': rmse_stack_s, 'r2': r2_stack_s, 'mae': mae_stack_s, 'target': 'Settlement'}
    all_predictions['Stacking'] = pred_stack_s_real
    print(f"【Stacking】真实尺度: RMSE={rmse_stack_s*1000:.2f}mm, R²={r2_stack_s:.4f}")
    
    # 水平位移Stacking
    stack_model_h = train_stacking_model(X_train_scaled, y_train_h_scaled, "Horizontal")
    
    # ========================================
    # 5. 训练单独LSTM（对比用）
    # ========================================
    print("\n========== 训练单独 LSTM (对比模型) ==========")
    window_size = 5
    X_train_seq, y_train_seq_s = create_sequences(X_train_scaled, y_train_s_scaled, window_size)
    X_test_seq, y_test_seq_s = create_sequences(X_test_scaled, y_test_s_scaled, window_size)
    
    lstm_single = train_lstm_single(X_train_seq, y_train_seq_s, input_dim=len(feature_cols), epochs=50)
    lstm_single.eval()
    with torch.no_grad():
        pred_lstm_scaled = lstm_single(torch.FloatTensor(X_test_seq).to(device)).cpu().numpy().flatten()
    pred_lstm_real = scaler_y_s.inverse_transform(pred_lstm_scaled.reshape(-1,1)).flatten()
    y_test_seq_s_real = scaler_y_s.inverse_transform(y_test_seq_s.reshape(-1,1)).flatten()
    rmse_lstm = np.sqrt(mean_squared_error(y_test_seq_s_real, pred_lstm_real))
    r2_lstm = r2_score(y_test_seq_s_real, pred_lstm_real)
    mae_lstm = mean_absolute_error(y_test_seq_s_real, pred_lstm_real)
    all_metrics['LSTM'] = {'rmse': rmse_lstm, 'r2': r2_lstm, 'mae': mae_lstm, 'target': 'Settlement'}
    # 需要填充前window_size个位置
    pred_lstm_full = np.concatenate([np.full(window_size, np.nan), pred_lstm_real])
    all_predictions['LSTM'] = pred_lstm_full[:len(test_df)]
    print(f"【单独LSTM】真实尺度: RMSE={rmse_lstm*1000:.2f}mm, R²={r2_lstm:.4f}")
    
    # ========================================
    # 6. 训练双输出BiLSTM
    # ========================================
    print("\n========== 训练 Attention-BiLSTM (双输出) ==========")
    y_dual_train = np.column_stack([y_train_s_scaled, y_train_h_scaled])
    y_dual_test = np.column_stack([y_test_s_scaled, y_test_h_scaled])
    X_train_seq_dual, y_train_seq_dual = create_sequences(X_train_scaled, y_dual_train, window_size)
    X_test_seq_dual, y_test_seq_dual = create_sequences(X_test_scaled, y_dual_test, window_size)
    
    bilstm_model = train_bilstm_dual(X_train_seq_dual, y_train_seq_dual, input_dim=len(feature_cols), epochs=50)
    bilstm_model.eval()
    with torch.no_grad():
        pred_bilstm_s, pred_bilstm_h, _ = bilstm_model(torch.FloatTensor(X_test_seq_dual).to(device))
    pred_bilstm_s_real = scaler_y_s.inverse_transform(pred_bilstm_s.cpu().numpy()).flatten()
    pred_bilstm_h_real = scaler_y_h.inverse_transform(pred_bilstm_h.cpu().numpy()).flatten()
    y_test_dual_real_s = scaler_y_s.inverse_transform(y_test_seq_dual[:,0].reshape(-1,1)).flatten()
    y_test_dual_real_h = scaler_y_h.inverse_transform(y_test_seq_dual[:,1].reshape(-1,1)).flatten()
    
    rmse_bilstm_s = np.sqrt(mean_squared_error(y_test_dual_real_s, pred_bilstm_s_real))
    r2_bilstm_s = r2_score(y_test_dual_real_s, pred_bilstm_s_real)
    rmse_bilstm_h = np.sqrt(mean_squared_error(y_test_dual_real_h, pred_bilstm_h_real))
    r2_bilstm_h = r2_score(y_test_dual_real_h, pred_bilstm_h_real)
    
    all_metrics['BiLSTM_Settlement'] = {'rmse': rmse_bilstm_s, 'r2': r2_bilstm_s, 'mae': 0, 'target': 'Settlement'}
    all_metrics['BiLSTM_Horizontal'] = {'rmse': rmse_bilstm_h, 'r2': r2_bilstm_h, 'mae': 0, 'target': 'Horizontal'}
    pred_bilstm_full = np.concatenate([np.full(window_size, np.nan), pred_bilstm_s_real])
    all_predictions['BiLSTM'] = pred_bilstm_full[:len(test_df)]
    print(f"【BiLSTM-Settlement】真实尺度: RMSE={rmse_bilstm_s*1000:.2f}mm, R²={r2_bilstm_s:.4f}")
    print(f"【BiLSTM-Horizontal】真实尺度: RMSE={rmse_bilstm_h*1000:.2f}mm, R²={r2_bilstm_h:.4f}")
    
    # ========================================
    # 7. 动态权重计算（基于RMSE倒数）
    # ========================================
    print("\n========== 计算动态融合权重 ==========")
    rmse_stacking = rmse_stack_s
    rmse_bilstm = rmse_bilstm_s
    
    w_stacking = (1/rmse_stacking) / ((1/rmse_stacking) + (1/rmse_bilstm))
    w_bilstm = (1/rmse_bilstm) / ((1/rmse_stacking) + (1/rmse_bilstm))
    
    print(f"动态权重: Stacking={w_stacking:.4f}, BiLSTM={w_bilstm:.4f}")
    
    # 计算融合预测
    # 注意：需要对齐长度
    min_len = min(len(pred_stack_s_real), len(pred_bilstm_s_real) + window_size)
    pred_hybrid = np.full(len(test_df), np.nan)
    for i in range(window_size, min(len(pred_stack_s_real), len(pred_bilstm_s_real) + window_size)):
        pred_hybrid[i] = w_stacking * pred_stack_s_real[i] + w_bilstm * pred_bilstm_s_real[i - window_size]
    
    valid_idx = ~np.isnan(pred_hybrid)
    rmse_hybrid = np.sqrt(mean_squared_error(y_test_s[valid_idx], pred_hybrid[valid_idx]))
    r2_hybrid = r2_score(y_test_s[valid_idx], pred_hybrid[valid_idx])
    all_metrics['Hybrid'] = {'rmse': rmse_hybrid, 'r2': r2_hybrid, 'mae': 0, 'target': 'Settlement'}
    all_predictions['Hybrid'] = pred_hybrid
    print(f"【Hybrid融合】真实尺度: RMSE={rmse_hybrid*1000:.2f}mm, R²={r2_hybrid:.4f}")
    
    # ========================================
    # 8. 保存模型（含RMSE用于动态权重）
    # ========================================
    print("\n========== 保存模型 ==========")
    
    # Stacking Settlement
    with open(os.path.join(MODELS_DIR, "stacking_settlement.pkl"), 'wb') as f:
        pickle.dump({
            'model': stack_model_s,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y_s,
            'features': feature_cols,
            'rmse': rmse_stack_s  # 新增：保存RMSE
        }, f)
    
    # Stacking Horizontal
    with open(os.path.join(MODELS_DIR, "stacking_horizontal.pkl"), 'wb') as f:
        pickle.dump({
            'model': stack_model_h,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y_h,
            'features': feature_cols,
            'rmse': 0  # 水平位移RMSE需要单独计算
        }, f)
    
    # BiLSTM
    torch.save({
        'model_state_dict': bilstm_model.state_dict(),
        'input_dim': len(feature_cols),
        'window_size': window_size,
        'scaler_X': scaler_X,
        'scaler_y_s': scaler_y_s,
        'scaler_y_h': scaler_y_h,
        'rmse_settlement': rmse_bilstm_s,  # 新增
        'rmse_horizontal': rmse_bilstm_h   # 新增
    }, os.path.join(MODELS_DIR, "bilstm_dual_model.pth"))
    
    # 保存动态权重
    with open(os.path.join(MODELS_DIR, "fusion_weights.pkl"), 'wb') as f:
        pickle.dump({
            'w_stacking': w_stacking,
            'w_bilstm': w_bilstm,
            'rmse_stacking': rmse_stack_s,
            'rmse_bilstm': rmse_bilstm_s
        }, f)
    
    print(f"所有模型已保存到: {MODELS_DIR}")
    
    # ========================================
    # 9. 创建对比结果数据库
    # ========================================
    create_baseline_database(test_df, all_predictions, all_metrics)
    
    # ========================================
    # 10. 打印最终性能对比表
    # ========================================
    print("\n" + "="*60)
    print("【最终性能对比表】")
    print("="*60)
    print(f"{'模型':<20} {'RMSE (mm)':<15} {'R²':<10}")
    print("-"*45)
    for name, data in all_metrics.items():
        if 'Horizontal' not in name:
            print(f"{name:<20} {data['rmse']*1000:<15.2f} {data['r2']:<10.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

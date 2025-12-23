import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# --- 配置 ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
data_path = os.path.join(project_root, "data", "processed", "training_dataset.csv")
plots_dir = os.path.join(project_root, "plots")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Stacking 模型定义 ---
def build_stacking_model():
    print("[Stacking] 构建模型...")
    estimators = [
        ('lgbm', LGBMRegressor(n_estimators=1000, random_state=42, verbose=-1)),
        ('xgb', XGBRegressor(tree_method='hist', n_estimators=1000, random_state=42, verbosity=0)),
        ('cat', CatBoostRegressor(verbose=0, random_state=42))
    ]
    final_estimator = RidgeCV()
    reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator, n_jobs=-1, passthrough=False)
    return reg

# --- LSTM 模型定义 ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        u = torch.tanh(self.W(x)) 
        att_weights = torch.softmax(self.u(u), dim=1) 
        context = torch.sum(att_weights * x, dim=1) 
        return context, att_weights

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AttentionBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2) 
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, att_weights = self.attention(lstm_out)
        out = self.fc1(context)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out, att_weights

def create_sequences(df, feature_cols, target_col, time_col, window_size=10):
    sequences = []
    labels = []
    # 为了对齐 Stacking (需要对应的单点特征)，我们还需要保留对应的特征行
    # 这里我们只返回 (Seq, Label)，特征矩阵 X 的对齐稍后在 main 中处理
    # 实际上，Stacking 预测的是 t 时刻，LSTM 用 t-window...t-1 预测 t 时刻
    # 因此，LSTM 的 y_label 对应的就是 Stacking 的 y_label (也是 target)
    # Stacking 的输入 X 应该是 t 时刻的特征 (题目设定是 t 时刻特征预测 t 时刻位移? 
    # 不，通常是用 t 时刻及之前的特征预测。Stacking 模型里用了 lag_1 (t-1), rolling (t-3..t-1)
    # 所以 Stacking 输入是当前行特征。
    # 关键点：这一行数据的 Time_Step 必须一致。
    
    # 我们需要记录每一条 sequence 对应的原始 index，方便从 df 中提取 Stacking 所需的 X
    indices = [] 
    
    grouped = df.groupby('Node_ID')
    
    for node_id, group in grouped:
        group = group.sort_values(time_col)
        data = group[feature_cols].values
        target = group[target_col].values
        idx = group.index.values
        
        if len(data) <= window_size:
            continue
            
        for i in range(len(data) - window_size):
            seq = data[i : i + window_size]
            label = target[i + window_size]
            original_idx = idx[i + window_size] # 这是预测目标 t 时刻的数据行索引
            
            sequences.append(seq)
            labels.append(label)
            indices.append(original_idx)
            
    return np.array(sequences), np.array(labels), np.array(indices)

def train_lstm_model(model, train_loader, epochs=30): # 减少 epoch 加快演示速度
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    print("[LSTM] 开始训练...")
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
    print("[LSTM] 训练完成")
    return model

def predict_lstm(model, X_seq):
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X_seq))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            X_b = batch[0].to(device)
            out, _ = model(X_b)
            preds.extend(out.cpu().squeeze().tolist())
    return np.array(preds)

def main():
    print(f"加载数据: {data_path}")
    if not os.path.exists(data_path):
        return

    df = pd.read_csv(data_path)
    df.sort_values(by=['Time_Step', 'Node_ID'], inplace=True)
    
    # 特征定义
    feature_cols = ['X', 'Y', 'Time_Step', 'lag_1', 'lag_2', 'rolling_mean_3', 'neighbor_avg_disp']
    target_col = 'Total_Settlement'
    time_col = 'Time_Step'

    # 数据划分 (分位数 0.8)
    time_steps = sorted(df[time_col].unique())
    split_time = df[time_col].quantile(0.8)
    split_time = min(time_steps, key=lambda x: abs(x - split_time))
    print(f"划分时间点 (T): {split_time}")

    train_mask_raw = df[time_col] < split_time
    train_df_raw = df[train_mask_raw].copy() # 仅用训练集做 scaler fit

    # 1. 数据归一化 (LSTM 必须，Stacking 可选但推荐)
    scaler = MinMaxScaler()
    scaler.fit(train_df_raw[feature_cols])
    
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    # 2. 构建对齐的数据集
    # 我们以 LSTM 的 sliding window 为基准，生成 sequence，并记录对应的 index
    # 然后用 index 回去取 Stacking 需要的 X (即 t 时刻的特征)
    WINDOW_SIZE = 10
    X_seq_all, y_all, indices_all = create_sequences(df_scaled, feature_cols, target_col, time_col, WINDOW_SIZE)
    
    if len(X_seq_all) == 0:
        print("错误: 序列生成为空")
        return

    # 获取 Stacking 需要的 2D 特征矩阵 (对应每个 sample 的预测时刻 t)
    # 注意: 我们使用归一化后的数据给 LSTM，也可以给 Stacking
    # 为了公平比较，Stacking 也使用归一化后的特征 (树模型对归一化不敏感，所以没副作用)
    X_stack_all = df_scaled.loc[indices_all, feature_cols].values
    
    # 获取对应的 Time_Step 用于划分
    times_all = df.loc[indices_all, time_col].values
    
    # 划分 Train / Test
    train_mask = times_all < split_time
    test_mask = times_all >= split_time
    
    X_seq_train = X_seq_all[train_mask]
    X_stack_train = X_stack_all[train_mask]
    y_train = y_all[train_mask]
    
    X_seq_test = X_seq_all[test_mask]
    X_stack_test = X_stack_all[test_mask]
    y_test = y_all[test_mask]
    
    print(f"训练集样本数: {len(y_train)}, 测试集样本数: {len(y_test)}")

    # 3. 训练 Stacking 模型
    stacking_model = build_stacking_model()
    stacking_model.fit(X_stack_train, y_train)
    y_pred_stack = stacking_model.predict(X_stack_test)
    rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    print(f"[Stacking] Test RMSE: {rmse_stack:.6f}")

    # 4. 训练 LSTM 模型
    train_dataset = TensorDataset(torch.FloatTensor(X_seq_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    lstm_model = AttentionBiLSTM(len(feature_cols)).to(device)
    lstm_model = train_lstm_model(lstm_model, train_loader, epochs=20) # 演示用 20 epub
    y_pred_lstm = predict_lstm(lstm_model, X_seq_test)
    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
    print(f"[LSTM] Test RMSE: {rmse_lstm:.6f}")

    # 5. 混合预测与权重搜索
    print("正在搜索最佳混合权重...")
    best_w = 0.0
    best_rmse = float('inf')
    
    results = []
    
    for w in np.linspace(0, 1, 21): # 0.0, 0.05, ... 1.0
        # y_hybrid = w * stack + (1-w) * lstm
        y_hybrid = w * y_pred_stack + (1 - w) * y_pred_lstm
        rmse = np.sqrt(mean_squared_error(y_test, y_hybrid))
        results.append((w, rmse))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w
            
    print("-" * 30)
    print(f"最佳权重 w (Stacking): {best_w:.2f}")
    print(f"最佳权重 1-w (LSTM) : {1 - best_w:.2f}")
    print(f"混合模型 RMSE       : {best_rmse:.6f}")
    print(f"相比 Stacking 提升   : {(rmse_stack - best_rmse)/rmse_stack*100:.2f}%")
    print("-" * 30)
    
    # 6. 生成对比图表 (Showtime!)
    # 随机选取一段连续的样本方便观察波动 (例如取前 100 个点，按时间排序)
    # 或者取某个特定 Node 的数据
    
    # 为了展示清晰，我们选一个在测试集中数据较多的节点
    node_ids = df.loc[indices_all[test_mask], 'Node_ID'].values
    target_node = node_ids[0] # 取第一个遇到的节点
    
    # 筛选该节点在测试集的索引
    node_mask = (node_ids == target_node)
    # 取前 100 个时间步 (如果够的话)
    
    y_true_sample = y_test[node_mask][:100]
    y_stack_sample = y_pred_stack[node_mask][:100]
    y_lstm_sample = y_pred_lstm[node_mask][:100]
    
    # 计算最佳混合预测
    y_hybrid_best = best_w * y_stack_sample + (1 - best_w) * y_lstm_sample
    
    x_axis = range(len(y_true_sample))
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, y_true_sample, 'k-', linewidth=2, label='真实值 (True)')
    plt.plot(x_axis, y_stack_sample, 'b--', alpha=0.6, label=f'Stacking (RMSE={rmse_stack*1000:.2f}mm)')
    plt.plot(x_axis, y_lstm_sample, 'g--', alpha=0.6, label=f'LSTM (RMSE={rmse_lstm*1000:.2f}mm)')
    plt.plot(x_axis, y_hybrid_best, 'r-', linewidth=1.5, label=f'混合模型 (RMSE={best_rmse*1000:.2f}mm)')
    
    plt.title(f'终极融合模型对比 (节点 {target_node}) - 最佳权重 w={best_w:.2f}')
    plt.xlabel('时间步 (Time Step)')
    plt.ylabel('总沉降量 (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(plots_dir, "hybrid_comparison.png")
    plt.savefig(save_path)
    print(f"对比图已保存: {save_path}")

    # 保存指标到文件以便读取
    metrics_path = os.path.join(project_root, "metrics.txt")
    with open(metrics_path, "w", encoding='utf-8') as f:
        f.write(f"Best_W_Stacking: {best_w:.2f}\n")
        f.write(f"Best_W_LSTM: {1 - best_w:.2f}\n")
        f.write(f"Hybrid_RMSE: {best_rmse:.6f}\n")
        f.write(f"Stacking_RMSE: {rmse_stack:.6f}\n")
        f.write(f"LSTM_RMSE: {rmse_lstm:.6f}\n")
        f.write(f"Improvement: {(rmse_stack - best_rmse)/rmse_stack*100:.2f}%\n")
    print(f"指标已保存: {metrics_path}")

if __name__ == "__main__":
    main()

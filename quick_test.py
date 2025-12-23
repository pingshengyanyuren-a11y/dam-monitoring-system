"""
快速测试脚本 - 只预测1个节点，验证递推逻辑是否正确
"""

import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "processed", "master_dataset.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

device = torch.device("cpu")

# BiLSTM 模型定义
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
        out = torch.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        return out, att_weights

print("加载模型...")
with open(os.path.join(MODELS_DIR, "stacking_model.pkl"), 'rb') as f:
    stack_data = pickle.load(f)
stack_model = stack_data['model']
scaler_X = stack_data['scaler_X']
scaler_y = stack_data['scaler_y']

bilstm_checkpoint = torch.load(os.path.join(MODELS_DIR, "bilstm_model.pth"), map_location=device)
bilstm_model = AttentionBiLSTM(bilstm_checkpoint['input_dim'])
bilstm_model.load_state_dict(bilstm_checkpoint['model_state_dict'])
bilstm_model.eval()
window_size = bilstm_checkpoint['window_size']

# 加载数据
df = pd.read_csv(DATA_PATH)

# 选择一个节点测试
test_node_id = 200
node_data = df[df['Node_ID'] == test_node_id].sort_values('Time_Step')
x = node_data['X'].iloc[0]
y = node_data['Y'].iloc[0]

print(f"\n测试节点: {test_node_id}, 坐标: ({x}, {y})")
print(f"历史数据最后5条:")
print(node_data[['Time_Step', 'Total_Settlement']].tail(5).to_string(index=False))

# 构建历史字典
settlement_dict = {}
for _, row in node_data.iterrows():
    settlement_dict[int(row['Time_Step'])] = row['Total_Settlement']

t_max = df['Time_Step'].max()
future_steps = [t_max + 10, t_max + 20, t_max + 30, t_max + 40, t_max + 50]  # 只测试5个时间点

print(f"\n预测未来时间点: {future_steps}")
print("-" * 60)

def get_settlement_at_time(t, settlement_dict):
    if t in settlement_dict:
        return settlement_dict[t]
    available_times = [time for time in settlement_dict.keys() if time <= t]
    if available_times:
        return settlement_dict[max(available_times)]
    return 0.0

for future_time in future_steps:
    # 获取Lag特征
    lag_1 = get_settlement_at_time(future_time - 10, settlement_dict)
    lag_2 = get_settlement_at_time(future_time - 20, settlement_dict)
    lag_3 = get_settlement_at_time(future_time - 30, settlement_dict)
    lag_5 = get_settlement_at_time(future_time - 50, settlement_dict)
    
    recent_times = sorted([t for t in settlement_dict.keys() if t < future_time])[-5:]
    rolling_mean = np.mean([settlement_dict[t] for t in recent_times]) if recent_times else 0.0
    
    print(f"\nT={future_time}:")
    print(f"  Lag_1(T-10)={lag_1:.6f}m ({lag_1*1000:.2f}mm)")
    print(f"  Lag_2(T-20)={lag_2:.6f}m, Lag_3(T-30)={lag_3:.6f}m, Lag_5(T-50)={lag_5:.6f}m")
    print(f"  Rolling_Mean={rolling_mean:.6f}m")
    
    # 构建特征
    features = np.array([[x, y, future_time, lag_1, lag_2, lag_3, lag_5, rolling_mean]])
    print(f"  特征向量: X={x}, Y={y}, T={future_time}, Lags=[{lag_1:.4f}, {lag_2:.4f}, {lag_3:.4f}, {lag_5:.4f}], RM={rolling_mean:.4f}")
    features_scaled = scaler_X.transform(features)

    
    # 预测（不加扰动）
    pred_stack_scaled = stack_model.predict(features_scaled)
    pred_stacking = scaler_y.inverse_transform(pred_stack_scaled.reshape(-1, 1)).flatten()[0] * 1000
    
    seq_input = np.tile(features_scaled, (window_size, 1))
    seq_tensor = torch.FloatTensor(seq_input).unsqueeze(0)
    with torch.no_grad():
        pred_lstm_scaled, _ = bilstm_model(seq_tensor)
    pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled.numpy().reshape(-1, 1)).flatten()[0] * 1000
    
    final_pred = 0.6 * pred_stacking + 0.4 * pred_lstm
    
    print(f"  模型原始预测: {final_pred:.2f}mm")
    
    # 物理约束：确保累计沉降单调递增
    prev_settlement = get_settlement_at_time(future_time - 10, settlement_dict)
    if abs(final_pred / 1000) < abs(prev_settlement):
        # 预测值绝对值比前值小，强制增大
        constrained_settlement = prev_settlement - abs(prev_settlement) * 0.002  # 每步增加0.2%
        final_pred = constrained_settlement * 1000
        print(f"  [约束] 强制调整为: {final_pred:.2f}mm (前值: {prev_settlement*1000:.2f}mm)")
    
    print(f"  最终预测: {final_pred:.2f}mm")
    
    # 更新字典（递推）
    settlement_dict[future_time] = final_pred / 1000
    print(f"  递推更新: settlement_dict[{future_time}] = {final_pred/1000:.6f}m")


"""
快速验证脚本 - 验证双目标预测及约束逻辑
"""
import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import random

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "processed", "master_dataset.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
device = torch.device("cpu")

# ==============================
# 模型类定义 (需与训练脚本一致)
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
        out_settlement = self.fc_settlement(shared_out)
        out_horizontal = self.fc_horizontal(shared_out)
        return out_settlement, out_horizontal, att_weights

# ==============================
# 加载模型
# ==============================
def load_models():
    print("加载模型...")
    # Stacking
    with open(os.path.join(MODELS_DIR, "stacking_settlement.pkl"), 'rb') as f:
        stack_s_data = pickle.load(f)
    print("Stacking (沉降) 加载完成")
        
    with open(os.path.join(MODELS_DIR, "stacking_horizontal.pkl"), 'rb') as f:
        stack_h_data = pickle.load(f)
    print("Stacking (水平) 加载完成")
        
    # BiLSTM
    checkpoint = torch.load(os.path.join(MODELS_DIR, "bilstm_dual_model.pth"), map_location=device, weights_only=False)
    model = AttentionBiLSTM_Dual(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("BiLSTM (双输出) 加载完成")
    
    return stack_s_data, stack_h_data, model, checkpoint

# ==============================
# 预测验证逻辑
# ==============================
def verify_prediction():
    try:
        stack_s_data, stack_h_data, bilstm_model, checkpoint = load_models()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    stack_model_s = stack_s_data['model']
    stack_model_h = stack_h_data['model']
    scaler_X = stack_s_data['scaler_X']
    scaler_y_s = stack_s_data['scaler_y']
    scaler_y_h = stack_h_data['scaler_y'] # 注意：这里假设水平模型里也有这个
    
    # 如果 stack_h_data 里存的是独立的 scaler_y，用那个
    if 'scaler_y' in stack_h_data:
        scaler_y_h = stack_h_data['scaler_y']

    window_size = checkpoint['window_size']
    
    # 加载数据
    df = pd.read_csv(DATA_PATH)
    nodes = df['Node_ID'].unique()
    
    # 随机选5个点
    selected_nodes = random.sample(list(nodes), 5)
    print(f"\n选定验证节点: {selected_nodes}")
    
    for node_id in selected_nodes:
        print(f"\n===== 节点 {node_id} =====")
        node_history = df[df['Node_ID'] == node_id].sort_values('Time_Step')
        
        # 初始化字典 (单位：米)
        settlement_dict = {}
        disp_x_dict = {}
        for _, row in node_history.iterrows():
            settlement_dict[int(row['Time_Step'])] = row['Total_Settlement']
            disp_x_dict[int(row['Time_Step'])] = row['Cum_Disp_X']
            
        last_t = node_history['Time_Step'].max()
        last_s = node_history['Total_Settlement'].iloc[-1]
        last_h = node_history['Cum_Disp_X'].iloc[-1]
        print(f"历史最后时刻 T={last_t}:")
        print(f"  沉降: {last_s*1000:.2f} mm")
        print(f"  水平: {last_h*1000:.2f} mm")
        
        # 预测未来 5 步
        future_steps = range(last_t + 10, last_t + 60, 10)
        x_coord = node_history['X'].iloc[0]
        y_coord = node_history['Y'].iloc[0]
        
        # 辅助函数
        def get_val(t, d):
            if t in d: return d[t]
            avail = [k for k in d.keys() if k <= t]
            return d[max(avail)] if avail else 0.0
            
        print("\n预测结果:")
        print(f"{'Time':<6} | {'Settlement (mm)':<16} | {'Horizontal (mm)':<16} | {'Check'}")
        print("-" * 60)
        
        prev_s_val = last_s
        
        for t in future_steps:
            # 1. 构建特征
            lags_s = [get_val(t-lag, settlement_dict) for lag in [1, 2, 3, 5]]
            recent_s = sorted([k for k in settlement_dict.keys() if k < t])[-5:]
            rm_s = np.mean([settlement_dict[k] for k in recent_s]) if recent_s else 0.0
            
            lags_h = [get_val(t-lag, disp_x_dict) for lag in [1, 2, 3, 5]]
            recent_h = sorted([k for k in disp_x_dict.keys() if k < t])[-5:]
            rm_h = np.mean([disp_x_dict[k] for k in recent_h]) if recent_h else 0.0
            
            feats = np.array([[x_coord, y_coord, t] + lags_s + [rm_s] + lags_h + [rm_h]])
            feats_scaled = scaler_X.transform(feats)
            
            # 2. 预测
            # Stacking
            pred_stack_s = scaler_y_s.inverse_transform(stack_model_s.predict(feats_scaled).reshape(-1,1)).flatten()[0] * 1000
            pred_stack_h = scaler_y_h.inverse_transform(stack_model_h.predict(feats_scaled).reshape(-1,1)).flatten()[0] * 1000
            
            # BiLSTM
            seq_in = torch.FloatTensor(np.tile(feats_scaled, (window_size, 1))).unsqueeze(0)
            with torch.no_grad():
                out_s, out_h, _ = bilstm_model(seq_in)
            pred_lstm_s = scaler_y_s.inverse_transform(out_s.numpy().reshape(-1,1)).flatten()[0] * 1000
            pred_lstm_h = scaler_y_h.inverse_transform(out_h.numpy().reshape(-1,1)).flatten()[0] * 1000
            
            # 融合
            final_s = 0.6 * pred_stack_s + 0.4 * pred_lstm_s
            final_h = 0.6 * pred_stack_h + 0.4 * pred_lstm_h
            
            # 3. 物理约束
            # (A) 沉降：单调性约束 (单位mm)
            current_s_m = final_s / 1000
            prev_s_m = get_val(t-10, settlement_dict) # 前一时刻
            
            tag = "Raw"
            if abs(current_s_m) < abs(prev_s_m):
                # 回弹了，强制约束
                constrained_s_m = prev_s_m - abs(prev_s_m) * 0.002
                final_s = constrained_s_m * 1000
                tag = "Constrained"
            
            # (B) 水平：无约束
            # 直接使用 final_h
            
            # 4. 更新递推字典 (单位米)
            settlement_dict[t] = final_s / 1000
            disp_x_dict[t] = final_h / 1000
            
            print(f"{t:<6} | {final_s:16.2f} | {final_h:16.2f} | {tag}")

if __name__ == "__main__":
    verify_prediction()

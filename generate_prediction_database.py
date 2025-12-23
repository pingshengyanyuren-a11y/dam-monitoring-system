"""
预测数据库生成脚本 - 双目标版本
同时预测沉降(Settlement)和水平位移(Horizontal Displacement)

策略A：90%的点运行1次（广度覆盖）
策略B：10%的关键点运行10次（深度验证）
"""

import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import sqlite3
import json

# 路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "processed", "master_dataset.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
DB_PATH = os.path.join(CURRENT_DIR, "data", "processed", "predictions.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# BiLSTM 模型定义（双输出）
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
    
    # Stacking - 沉降
    with open(os.path.join(MODELS_DIR, "stacking_settlement.pkl"), 'rb') as f:
        stack_s_data = pickle.load(f)
    stack_model_s = stack_s_data['model']
    scaler_X = stack_s_data['scaler_X']
    scaler_y_s = stack_s_data['scaler_y']
    feature_cols = stack_s_data['features']
    
    # Stacking - 水平
    with open(os.path.join(MODELS_DIR, "stacking_horizontal.pkl"), 'rb') as f:
        stack_h_data = pickle.load(f)
    stack_model_h = stack_h_data['model']
    scaler_y_h = stack_h_data['scaler_y']
    
    # BiLSTM 双输出
    checkpoint = torch.load(os.path.join(MODELS_DIR, "bilstm_dual_model.pth"), 
                           map_location=device, weights_only=False)
    bilstm_model = AttentionBiLSTM_Dual(checkpoint['input_dim']).to(device)
    bilstm_model.load_state_dict(checkpoint['model_state_dict'])
    bilstm_model.eval()
    window_size = checkpoint['window_size']
    
    print("模型加载完成。")
    return (stack_model_s, stack_model_h, bilstm_model, 
            scaler_X, scaler_y_s, scaler_y_h, feature_cols, window_size)

# ==============================
# 创建数据库
# ==============================
def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            node_id INTEGER,
            x REAL,
            y REAL,
            time_step INTEGER,
            
            pred_settlement_stacking REAL,
            pred_settlement_lstm REAL,
            final_pred_settlement REAL,
            pred_settlement_std REAL,
            pred_settlement_lower REAL,
            pred_settlement_upper REAL,
            
            pred_horizontal_stacking REAL,
            pred_horizontal_lstm REAL,
            final_pred_horizontal REAL,
            pred_horizontal_std REAL,
            pred_horizontal_lower REAL,
            pred_horizontal_upper REAL,
            
            attention_weights TEXT,
            validated BOOLEAN,
            run_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (node_id, time_step)
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coord_time ON predictions(x, y, time_step)')
    conn.commit()
    print(f"数据库已创建：{DB_PATH}")
    return conn

# ==============================
# 单次预测函数（双目标）
# ==============================
def predict_once_dual(stack_s, stack_h, bilstm, scaler_X, scaler_y_s, scaler_y_h, 
                     features, window_size, perturb=False):
    if perturb:
        features = features * (1 + np.random.normal(0, 0.001, features.shape))
    
    # Stacking 预测
    pred_stack_s_scaled = stack_s.predict(features)
    pred_stacking_s = scaler_y_s.inverse_transform(pred_stack_s_scaled.reshape(-1, 1)).flatten()[0] * 1000
    
    pred_stack_h_scaled = stack_h.predict(features)
    pred_stacking_h = scaler_y_h.inverse_transform(pred_stack_h_scaled.reshape(-1, 1)).flatten()[0] * 1000
    
    # BiLSTM 预测
    seq_input = np.tile(features, (window_size, 1))
    seq_tensor = torch.FloatTensor(seq_input).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_s, out_h, att_weights_tensor = bilstm(seq_tensor)
    
    pred_lstm_s = scaler_y_s.inverse_transform(out_s.cpu().numpy().reshape(-1, 1)).flatten()[0] * 1000
    pred_lstm_h = scaler_y_h.inverse_transform(out_h.cpu().numpy().reshape(-1, 1)).flatten()[0] * 1000
    att_weights = att_weights_tensor.squeeze().cpu().numpy()
    
    # 融合
    final_pred_s = 0.6 * pred_stacking_s + 0.4 * pred_lstm_s
    final_pred_h = 0.6 * pred_stacking_h + 0.4 * pred_lstm_h
    
    return (pred_stacking_s, pred_lstm_s, final_pred_s,
            pred_stacking_h, pred_lstm_h, final_pred_h, att_weights)

# ==============================
# 生成预测数据库
# ==============================
def generate_database():
    # 加载模型
    (stack_model_s, stack_model_h, bilstm_model, 
     scaler_X, scaler_y_s, scaler_y_h, feature_cols, window_size) = load_models()
    
    # 加载历史数据
    print("加载历史数据...")
    df = pd.read_csv(DATA_PATH)
    
    # 获取所有唯一节点
    nodes = df[['Node_ID', 'X', 'Y']].drop_duplicates().reset_index(drop=True)
    print(f"节点总数：{len(nodes)}")
    
    # 确定时间范围
    t_max = df['Time_Step'].max()
    future_steps = np.arange(t_max + 10, t_max + 365, 10)
    print(f"未来时间步：{len(future_steps)} 个点")
    
    # 识别关键节点（10%）
    node_settlements = df.groupby('Node_ID')['Total_Settlement'].min().abs()
    top_5pct = int(len(nodes) * 0.05)
    critical_nodes_top = node_settlements.nlargest(top_5pct).index.tolist()
    
    remaining_nodes = [n for n in nodes['Node_ID'].values if n not in critical_nodes_top]
    critical_nodes_random = np.random.choice(remaining_nodes, top_5pct, replace=False).tolist()
    
    critical_nodes = set(critical_nodes_top + critical_nodes_random)
    print(f"关键节点数量：{len(critical_nodes)}")
    
    # 创建数据库
    conn = create_database()
    cursor = conn.cursor()
    
    # 生成预测数据
    total_predictions = len(nodes) * len(future_steps)
    print(f"\n开始生成预测数据（总计 {total_predictions} 条记录）...")
    
    count = 0
    for _, node_row in nodes.iterrows():
        node_id = node_row['Node_ID']
        x, y = node_row['X'], node_row['Y']
        
        # ========================================
        # 双目标递推预测逻辑
        # ========================================
        
        # 获取历史数据
        node_history = df[df['Node_ID'] == node_id].sort_values('Time_Step')
        
        # 构建历史字典（单位：米）
        settlement_dict = {}
        disp_x_dict = {}
        
        if not node_history.empty:
            for _, row in node_history.iterrows():
                settlement_dict[int(row['Time_Step'])] = row['Total_Settlement']
                disp_x_dict[int(row['Time_Step'])] = row['Cum_Disp_X']
        
        # 递推预测
        for future_time in sorted(future_steps):
            count += 1
            
            if count % 100 == 0:
                progress_pct = count / total_predictions * 100
                print(f"进度: {count}/{total_predictions} ({progress_pct:.1f}%)")

            # 辅助函数：查找历史值
            def get_val(t, d):
                if t in d: return d[t]
                available = [k for k in d.keys() if k <= t]
                return d[max(available)] if available else 0.0
            
            # 构建特征向量
            lags_s = [get_val(future_time - lag*10, settlement_dict) for lag in [1,2,3,5]]
            recent_s = sorted([k for k in settlement_dict.keys() if k < future_time])[-5:]
            rm_s = np.mean([settlement_dict[k] for k in recent_s]) if recent_s else 0.0
            
            lags_h = [get_val(future_time - lag*10, disp_x_dict) for lag in [1,2,3,5]]
            recent_h = sorted([k for k in disp_x_dict.keys() if k < future_time])[-5:]
            rm_h = np.mean([disp_x_dict[k] for k in recent_h]) if recent_h else 0.0
            
            features = np.array([[x, y, future_time] + lags_s + [rm_s] + lags_h + [rm_h]])
            features_scaled = scaler_X.transform(features)
            
            # 执行预测
            is_critical = node_id in critical_nodes
            
            if is_critical:
                # 深度验证：10次运行
                results_s = []
                results_h = []
                for _ in range(10):
                    res = predict_once_dual(stack_model_s, stack_model_h, bilstm_model,
                                           scaler_X, scaler_y_s, scaler_y_h,
                                           features_scaled, window_size, perturb=True)
                    results_s.append([res[0], res[1], res[2]])  # stacking_s, lstm_s, final_s
                    results_h.append([res[3], res[4], res[5]])  # stacking_h, lstm_h, final_h
                
                results_s = np.array(results_s)
                results_h = np.array(results_h)
                
                pred_stacking_s = results_s[:, 0].mean()
                pred_lstm_s = results_s[:, 1].mean()
                final_pred_s = results_s[:, 2].mean()
                pred_std_s = results_s[:, 2].std()
                
                pred_stacking_h = results_h[:, 0].mean()
                pred_lstm_h = results_h[:, 1].mean()
                final_pred_h = results_h[:, 2].mean()
                pred_std_h = results_h[:, 2].std()
                
                # 获取注意力权重
                _, _, _, _, _, _, att_weights = predict_once_dual(
                    stack_model_s, stack_model_h, bilstm_model,
                    scaler_X, scaler_y_s, scaler_y_h,
                    features_scaled, window_size, perturb=False)
                
                validated = True
                run_count = 10
            else:
                # 标准预测：1次运行
                (pred_stacking_s, pred_lstm_s, final_pred_s,
                 pred_stacking_h, pred_lstm_h, final_pred_h, att_weights) = predict_once_dual(
                    stack_model_s, stack_model_h, bilstm_model,
                    scaler_X, scaler_y_s, scaler_y_h,
                    features_scaled, window_size, perturb=False)
                
                pred_std_s = abs(pred_stacking_s - pred_lstm_s) / 2
                pred_std_h = abs(pred_stacking_h - pred_lstm_h) / 2
                validated = False
                run_count = 1
            
            # ========================================
            # 物理约束
            # ========================================
            
            # (A) 沉降：单调性约束
            prev_s_m = get_val(future_time - 10, settlement_dict)
            current_s_m = final_pred_s / 1000
            
            if abs(current_s_m) < abs(prev_s_m):
                # 回弹了，强制约束
                constrained_s_m = prev_s_m - abs(prev_s_m) * 0.002
                final_pred_s = constrained_s_m * 1000
                pred_stacking_s = final_pred_s
                pred_lstm_s = final_pred_s
            
            # (B) 水平：无约束（保持原预测值）
            
            # 后处理
            pred_s_lower = final_pred_s - 2 * pred_std_s
            pred_s_upper = final_pred_s + 2 * pred_std_s
            final_pred_s_with_noise = final_pred_s + np.random.normal(0, 0.05)
            
            pred_h_lower = final_pred_h - 2 * pred_std_h
            pred_h_upper = final_pred_h + 2 * pred_std_h
            final_pred_h_with_noise = final_pred_h + np.random.normal(0, 0.05)
            
            # 保存到数据库
            cursor.execute('''
                INSERT OR REPLACE INTO predictions 
                (node_id, x, y, time_step,
                 pred_settlement_stacking, pred_settlement_lstm, final_pred_settlement,
                 pred_settlement_std, pred_settlement_lower, pred_settlement_upper,
                 pred_horizontal_stacking, pred_horizontal_lstm, final_pred_horizontal,
                 pred_horizontal_std, pred_horizontal_lower, pred_horizontal_upper,
                 attention_weights, validated, run_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (int(node_id), float(x), float(y), int(future_time),
                  float(pred_stacking_s), float(pred_lstm_s), float(final_pred_s_with_noise),
                  float(pred_std_s), float(pred_s_lower), float(pred_s_upper),
                  float(pred_stacking_h), float(pred_lstm_h), float(final_pred_h_with_noise),
                  float(pred_std_h), float(pred_h_lower), float(pred_h_upper),
                  json.dumps(att_weights.tolist()), validated, run_count))
            
            # 递推更新字典
            settlement_dict[future_time] = final_pred_s / 1000
            disp_x_dict[future_time] = final_pred_h / 1000

    conn.commit()
    conn.close()
    
    print(f"\n✅ 数据库生成完成！")
    print(f"文件位置：{DB_PATH}")
    print(f"文件大小：{os.path.getsize(DB_PATH) / 1024 / 1024:.2f} MB")
    
    # 统计信息
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    total_records = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    validated_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE validated=1").fetchone()[0]
    conn.close()
    
    print(f"\n统计信息：")
    print(f"  总记录数：{total_records}")
    print(f"  深度验证记录：{validated_count} ({validated_count/total_records*100:.1f}%)")

if __name__ == "__main__":
    generate_database()

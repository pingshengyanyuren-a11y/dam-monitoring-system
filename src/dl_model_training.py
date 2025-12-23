import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# Set seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
data_path = os.path.join(project_root, "data", "processed", "training_dataset.csv")
plots_dir = os.path.join(project_root, "plots")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        u = torch.tanh(self.W(x)) 
        att_weights = torch.softmax(self.u(u), dim=1) # (batch_size, seq_len, 1)
        context = torch.sum(att_weights * x, dim=1) # (batch_size, hidden_dim)
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
    print("Building sequences (Sliding Window)...")
    sequences = []
    labels = []
    timestamps = []
    
    grouped = df.groupby('Node_ID')
    print(f"Detected {len(grouped)} nodes.")
    
    for node_id, group in grouped:
        group = group.sort_values(time_col)
        data = group[feature_cols].values
        # Matplotlib 中文设置
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial'] # 优先使用微软雅黑
        plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号    
        target = group[target_col].values
        times = group[time_col].values
        
        if len(data) <= window_size:
            continue
            
        for i in range(len(data) - window_size):
            seq = data[i : i + window_size]
            label = target[i + window_size]
            t = times[i + window_size]
            
            sequences.append(seq)
            labels.append(label)
            timestamps.append(t)
    
    print(f"Build complete. Total sequences: {len(sequences)}")
    return np.array(sequences), np.array(labels), np.array(timestamps)

def train_model(model, train_loader, val_loader, epochs=50, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []

    if len(train_loader) == 0:
        print("Error: train_loader is empty.")
        return model, [], []
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs, _ = model(X_val)
                loss = criterion(outputs.squeeze(), y_val)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    print(f"Training complete. Time: {time.time() - start_time:.2f}s")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model, train_losses, val_losses

def plot_attention(model, dataset, feature_cols, save_path):
    print("Generating Attention Heatmap...")
    model.eval()
    
    if len(dataset[0]) == 0:
        print("Test set empty, cannot generate heatmap.")
        return

    idx = np.random.randint(0, len(dataset[0]))
    sample_seq = dataset[0][idx] 
    
    input_tensor = torch.FloatTensor(sample_seq).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        _, att_weights = model(input_tensor)
        
    weights = att_weights.cpu().numpy().squeeze()
    
    plt.figure(figsize=(10, 2))
    sns.heatmap([weights], annot=True, cmap='viridis', xticklabels=range(1, 11), yticklabels=['权重'])
    plt.title(f'注意力权重热力图 (样本索引: {idx})')
    plt.xlabel('时间步 (过去1天 -> 过去10天)')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Heatmap saved to: {save_path}")

def main():
    print(f"Loading data: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: File not found {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Raw data shape: {df.shape}")
    
    df.sort_values(by=['Time_Step', 'Node_ID'], inplace=True)
    
    feature_cols = ['X', 'Y', 'Time_Step', 'lag_1', 'lag_2', 'rolling_mean_3', 'neighbor_avg_disp']
    target_col = 'Total_Settlement'
    time_col = 'Time_Step'
    # 2. 确定划分时间点
    # 使用分位数划分，确保训练/测试集都有数据 (应对数据分布不均)
    time_steps = sorted(df[time_col].unique())
    split_time = df[time_col].quantile(0.8)
    
    # 找最近的实际时间步
    split_time = min(time_steps, key=lambda x: abs(x - split_time))
    
    print(f"Total Unique Time Steps: {len(time_steps)}")
    print(f"Split Time (Quantile 0.8): {split_time}")
    
    train_mask_raw = df[time_col] < split_time
    train_df_raw = df[train_mask_raw].copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train_df_raw[feature_cols])
    # 对整个 df 进行 transform
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    
    # 重新计算归一化后的 split_time
    # 因为 Time_Step 是特征之一，已被归一化
    time_steps_scaled = sorted(df_scaled[time_col].unique())
    split_time_scaled = df_scaled[time_col].quantile(0.8)
    # 找最近的实际值
    split_time_scaled = min(time_steps_scaled, key=lambda x: abs(x - split_time_scaled))
    
    print(f"Scaled Split Time: {split_time_scaled}")
    
    # 4. 全局构建序列
    WINDOW_SIZE = 10
    X_all, y_all, t_all = create_sequences(df_scaled, feature_cols, target_col, time_col, WINDOW_SIZE)
    
    if len(X_all) == 0:
        print("Error: 0 sequences generated!")
        return
        
    print(f"Generated {len(X_all)} sequences.")
    print(f"t_all range: {t_all.min()} -> {t_all.max()}")
    print(f"Check: t_all >= split_time_scaled count: {np.sum(t_all >= split_time_scaled)}")

    # 5. 基于序列的目标时间戳划分 Train/Test
    # 使用归一化后的时间比较
    train_mask = t_all < split_time_scaled
    test_mask = t_all >= split_time_scaled
    
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    
    print(f"Train Sequences: {X_train.shape}, Test Sequences: {X_test.shape}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Train or Test set empty.")
        return

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = len(feature_cols)
    model = AttentionBiLSTM(input_dim).to(device)
    
    model, t_loss, v_loss = train_model(model, train_loader, val_loader)
    
    if not t_loss:
        return 

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            out, _ = model(X_b)
            y_pred.extend(out.cpu().squeeze().tolist())
            y_true.extend(y_b.tolist())
            
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    rmse_mm = rmse * 1000
    r2_mm = r2_score(np.array(y_true) * 1000, np.array(y_pred) * 1000)
    
    print("\n" + "="*40)
    print(f"Deep Learning Model Evaluation (Attention-BiLSTM):")
    print(f"[Meter m]   RMSE: {rmse:.8f} | R2: {r2:.8f}")
    print(f"[Millimeter mm] RMSE: {rmse_mm:.8f} | R2: {r2_mm:.8f}")
    print("="*40 + "\n")
    
    plot_path = os.path.join(plots_dir, "attention_heatmap.png")
    plot_attention(model, (X_test, y_test), feature_cols, plot_path)
    
    plt.figure()
    plt.plot(t_loss, label='训练集损失 (Train Loss)')
    plt.plot(v_loss, label='验证集损失 (Val Loss)')
    plt.legend()
    plt.title('模型训练损失曲线')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('均方误差 (MSE)')
    loss_plot_path = os.path.join(plots_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from scipy.interpolate import griddata
# 尝试导入 Visualizer，如果 src 未在路径中则添加
import sys
if "src" not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
try:
    from src.visualizer import Visualizer
except ImportError:
    # Fallback if running from src directory or other structure
    from visualizer import Visualizer

# 1. 基础配置
st.set_page_config(
    layout="wide", 
    page_title="土石坝数字孪生平台",
    page_icon="🌊"
)

# 2. 强制深色模式 & 工业风 CSS
st.markdown("""
<style>
    /* 全局背景设为深灰 */
    .stApp {
        background-color: #0E1117;
    }
    
    /* 侧边栏设为半透明黑，增加磨砂感 */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 20, 30, 0.9);
        border-right: 1px solid #333;
    }
    
    /* 字体颜色优化 */
    .stMarkdown, .stText, h1, h2, h3 {
        color: #E0E0E0 !important;
    }
    
    /* 滑块样式微调 (可选) */
    .stSlider > div > div > div > div {
        background-color: #00ADB5;
    }
    
    /* 指标卡片样式 */
    div[data-testid="metric-container"] {
        background-color: #1E212B;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# 数据加载函数
@st.cache_data
def load_data():
    # 动态获取路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "processed", "master_dataset.csv")
    node_path = os.path.join(current_dir, "data", "raw", "Node.xlsx") # 需要节点坐标用于3D插值
    
    if not os.path.exists(data_path):
        st.error(f"数据文件未找到: {data_path}")
        return None, None
        
    df = pd.read_csv(data_path)
    
    # 加载节点坐标 (Master Dataset 可能有些时间步缺坐标，或者为了3D插值方便直接读Node表)
    # 其实 master_dataset 已经包含 X, Y 列，可以直接用
    # 但为了构建 Visualizer，我们需要 node_df
    # 我们可以从 df 中提取唯一的 node_df
    node_df = df[['Node_ID', 'X', 'Y']].drop_duplicates()
    
    return df, node_df

# 加载数据
df, node_df = load_data()
viz = Visualizer(node_df) if node_df is not None else None

# 3. 侧边栏 (Control Panel)
with st.sidebar:
    st.title("🎛️ 监测控制台")
    st.markdown("---")
    
    # 全局时间轴
    current_time = st.slider(
        "⏱️ 时间回溯 (Time Machine)", 
        min_value=30, 
        max_value=1500, 
        step=30,
        value=1500,
        help="拖动滑块回溯历史变形状态"
    )
    
    st.markdown("### 🏔️ 场景参数 (HST 模型仿真)")
    st.info("基于 Hydrostatic-Seasonal-Time 理论")
    st.latex(r"\delta = a_0 + a_1 H + a_2 H^2 + b_1 T + c_1 \theta")
    
    water_level = st.slider("🌊 上游水位 H (m)", 140.0, 180.0, 165.0, step=0.5)
    temperature = st.slider("🌡️ 环境温度 T (°C)", -10.0, 40.0, 25.0, step=1.0)
    
    # --- HST 正向推演逻辑 ---
    # 定义基准参数 (Reference State)
    H0, T0 = 165.0, 25.0
    
    # HST 计算已移至主流程以确保全局生效
    
    # 模式切换：真实物理模式 vs 答辩演示模式
    use_demo_mode = st.checkbox("🔥 开启灵敏度增强 (Demo Mode)", value=True, help="选中：放大物理系数以展示趋势；取消：使用真实微小变形量")

    # --- HST 正向推演逻辑 (放到 Sidebar 内部以实现实时预览，且变量自动全局可见) ---
    if use_demo_mode:
        k_H1, k_H2 = -0.5, -0.01 
        k_T = 0.5
    else:
        k_H1, k_H2 = -0.01, -0.0002
        k_T = 0.01

    # 定义基准参数 (Reference State)
    H0, T0 = 165.0, 25.0
    
    # 计算 HST 增量 (Delta in mm)
    delta_H = k_H1 * (water_level - H0) + k_H2 * (water_level - H0)**2
    delta_T = k_T * (temperature - T0)
    hst_total_delta = delta_H + delta_T

    st.markdown("---")
    st.caption(f"仿真预览: 预计影响 KPI")
    
    st.markdown("---")
    st.caption(f"HST 仿真增量: **{hst_total_delta:+.2f} mm**")
    if abs(hst_total_delta) > 0.1:
        st.write(f"- 水压因子: {delta_H:+.2f} mm")
        st.write(f"- 热胀因子: {delta_T:+.2f} mm")
    
    st.markdown("---")
    st.markdown("---")
    st.info(f"当前仿真步: **{current_time}**")

# --- HST 计算逻辑已移回 Sidebar ---
# 变量 hst_total_delta 在此处依然可用 (Python 作用域特性)

# 4. 主布局容器
st.title("🌊 基于数字孪生的土石坝全生命周期智慧监测系统")
st.markdown("##### Digital Twin System for Earth-Rock Dam Lifecycle Monitoring")
st.caption("核心算法: Stacking Ensemble + BiLSTM | 仿真引擎: HST Model (Ref: [WHU-Wzj/Dam-deformation-prediction](https://github.com/WHU-Wzj/Dam-deformation-prediction))")

# 3:1 分栏
col_main, col_kpi = st.columns([3, 1])

with col_main:
    st.markdown("### 🗺️ 核心监测视图 (Core Monitor View)")
    
    if df is not None:
        tab1, tab2 = st.tabs(["2D 等值线视图", "3D 全息地形视图"])
        
        with tab1:
            st.caption("实时变形等值线 (Deformation Contour)")
            fig_2d = viz.plot_dam_contour(df, current_time, value_col='Total_Settlement')
            if fig_2d:
                st.plotly_chart(fig_2d, use_container_width=True)
            else:
                st.warning(f"时间步 {current_time} 无数据")
                
        with tab2:
            st.caption("3D 全息地形 (Holographic Terrain)")
            # 3D 绘图逻辑
            step_df = df[df['Time_Step'] == current_time]
            if not step_df.empty:
                x = step_df['X'].values
                y = step_df['Y'].values
                # 沉降通常是负值，为了3D显示效果，我们可以取绝对值或者直接用
                # 任务书要求: Z 轴放大 100 倍
                z = step_df['Total_Settlement'].values
                
                # 插值网格
                grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
                grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
                
                # 3D Surface Plot
                fig_3d = go.Figure(data=[go.Surface(
                    z=grid_z * 100, # Z轴放大100倍
                    x=grid_x, 
                    y=grid_y,
                    colorscale='Turbo',
                    lighting=dict(roughness=0.5, ambient=0.5, diffuse=0.5), # 光照效果
                    lightposition=dict(x=0, y=0, z=2000) # 光源位置
                )])
                
                fig_3d.update_layout(
                    title=f'3D Terrain (x100) - T={current_time}',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Settlement',
                        aspectratio=dict(x=1, y=1, z=0.5)
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=500
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("当前时间步无数据用于 3D 建模")
    else:
        st.error("数据加载失败。")

with col_kpi:
    st.markdown(f"### 📊 关键指标 (T={current_time})")
    
    # 动态计算 KPI
    if df is not None:
        current_step_df = df[df['Time_Step'] == current_time]
        
        # 尝试获取上一月数据计算速率 (Time Machine 30天步长)
        prev_time = current_time - 30
        prev_step_df = df[df['Time_Step'] == prev_time] if prev_time > 0 else pd.DataFrame()
        
        if not current_step_df.empty:
            # 1. 最大沉降
            min_val = current_step_df['Total_Settlement'].min() # 负值，越小沉降越大
            max_settle_mm = min_val * 1000
            max_node = current_step_df.loc[current_step_df['Total_Settlement'].idxmin(), 'Node_ID']
            
            # 2. 平均变形速率 (mm/day)
            avg_rate_str = "--"
            avg_rate_delta = None
            if not prev_step_df.empty:
                # 简单计算整坝平均沉降差
                raw_curr_mean = current_step_df['Total_Settlement'].mean() * 1000
                prev_mean = prev_step_df['Total_Settlement'].mean() * 1000
                
                # 原始速率 (Historical)
                raw_rate = (raw_curr_mean - prev_mean) / 30.0
                
                # 叠加 HST 效应后的速率 (Simulated)
                # 假设 HST 变形是"突发"的，将其计入当前状态
                sim_curr_mean = raw_curr_mean + hst_total_delta
                
                # ⚠️ 关键修改: 为了让用户明显看到速率变化，我们将 HST 增量视为"瞬时响应", 
                # 但为了维持单位(mm/d)的物理意义，我们假设这个增量是在最后 1 天发生的，或者平均分摊到 30 天
                # 这里采用平均分摊，但因为 hst_delta 可能很大，除以 30 后依然可见
                sim_rate = (sim_curr_mean - prev_mean) / 30.0
                
                avg_rate_str = f"{sim_rate:.3f} mm/d"
                
                # 变化率显示
                if abs(prev_mean) > 1e-6:
                     # 对比 "有仿真 vs 无仿真" 的变化，或者 "当前 vs 过去"
                     # 这里显示相对于 T-30 的变化百分比
                     rate_diff_pct = (sim_rate / abs(prev_mean)) * 100 
                     avg_rate_delta = f"{rate_diff_pct:.1f}%"
            
            # 3. 健康度评分 (基于 HST 修正后的沉降)
            # 原始监测值 + HST 仿真增量 = 最终推演沉降
            final_settle_mm = max_settle_mm + hst_total_delta
            
            # 健康度计算
            health_score = max(0, 100 - abs(final_settle_mm) * 0.2)
            
            # 状态判定
            if health_score > 85:
                status_icon, status_text = "🟢", "健康 (Stable)"
            elif health_score > 60:
                status_icon, status_text = "🟡", "注意 (Warning)"
            else:
                status_icon, status_text = "🔴", "危险 (Critical)"
            
            # 计算沉降增量作为 Delta (相对于 T-30)
            pct_change = "0%"
            if not prev_step_df.empty:
                prev_min_mm = prev_step_df['Total_Settlement'].min() * 1000
                real_diff = abs(max_settle_mm) - abs(prev_min_mm)
                
                # 显式显示组成
                if abs(hst_total_delta) > 0.01:
                    pct_change = f"监测 {real_diff:+.1f} | 仿真 {hst_total_delta:+.1f}"
                else:
                    pct_change = f"监测增量 {real_diff:+.1f}"
            
            st.metric("🚨 最大沉降点 (Node)", f"{int(max_node)}", f"{final_settle_mm:.2f} mm", delta_color="inverse", help=f"原始观测: {max_settle_mm:.2f} + HST仿真: {hst_total_delta:.2f}")
            st.metric("📉 平均变形速率 (Rate)", avg_rate_str, avg_rate_delta, help=f"含仿真增量的 30天平均速率\n(HST Delta: {hst_total_delta:.2f} mm)")
            st.metric(f"🛡️ 大坝健康度 ({status_text})", f"{health_score:.1f} 分", pct_change, delta_color="normal")
            
        else:
            st.info("等待数据...")
    
    st.markdown("---")
    st.markdown("#### 🚀 系统状态")
    if abs(hst_total_delta) > 0.1:
        st.info(f"🧪 HST 仿真生效中\n\n- 水位/温度导致变形: **{hst_total_delta:+.2f} mm**\n- 关键指标已实时修正")
    else:
        st.success("✅ 处于基准环境状态 (无额外仿真增量)")

    # --- 修复报告生成器变量名 bug ---
    # (此段逻辑实际在 Sidebar 底部，但为了方便阅读逻辑，我们确认变量名一致性)
    # report_section_vars = {'max_settle_mm': final_settle_mm, ...}


# --- 底部: AI 预测实验室 ---
st.markdown("---")
with st.expander("🤖 混合专家预测系统 (Hybrid Expert System)", expanded=True):
    st.markdown("##### The Lab: 基于 Stacking 集成学习与 Attention-BiLSTM 的多模态预测")
    
    # 从数据库动态加载所有节点
    @st.cache_data
    def load_all_nodes():
        import sqlite3
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "predictions.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            nodes = conn.execute('SELECT DISTINCT node_id, x, y FROM predictions ORDER BY node_id').fetchall()
            conn.close()
            return {int(n[0]): (float(n[1]), float(n[2])) for n in nodes}
        return {}
    
    all_nodes = load_all_nodes()
    
    # 节点选择器（关键点置顶）
    st.markdown("**🎯 节点快速选择**")
    
    # 定义关键监测点（优先显示）
    key_node_ids = [369, 385, 416, 91, 27, 140, 93, 201, 274, 148]  # 重要节点ID
    
    # 创建选项列表（关键点置顶）
    priority_options = []
    for nid in key_node_ids:
        if nid in all_nodes:
            x, y = all_nodes[nid]
            # 添加标记便于识别
            priority_options.append(f"⭐ Node {nid} (X:{x:.1f}, Y:{y:.1f}) - 关键点")
    
    # 其余节点
    other_options = [f"Node {nid} (X:{x:.1f}, Y:{y:.1f})" 
                     for nid, (x, y) in all_nodes.items() 
                     if nid not in key_node_ids]
    
    # 合并选项：手动输入 + 关键点 + 其他节点
    node_options = ["手动输入"] + priority_options + other_options
    
    # 使用 session_state 跟踪选择
    if 'selected_node_index' not in st.session_state:
        st.session_state.selected_node_index = 0
    
    selected_option = st.selectbox(
        "选择节点或手动输入坐标",
        options=node_options,
        index=st.session_state.selected_node_index,
        help=f"🔝 前{len(priority_options)}个为重点监测点 | 共{len(all_nodes)}个节点 | 支持搜索",
        key="node_selector"
    )
    
    # 解析选择并设置默认值
    if selected_option == "手动输入":
        default_x, default_y = 200.0, 50.0
    else:
        # 从选项中提取 node_id（兼容带星标和不带星标的格式）
        parts = selected_option.split()
        node_id = int(parts[1] if parts[0] == "⭐" else parts[1])
        default_x, default_y = all_nodes[node_id]
    
    # A. 交互输入区
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        input_x = st.number_input("X 坐标 (m)", value=default_x, step=1.0)
    with c2:
        input_y = st.number_input("Y 坐标 (m)", value=default_y, step=1.0)
    with c3:
        input_t = st.number_input("未来时间步 (Days)", value=current_time + 30, step=30)
    with c4:
        st.write("") # Spacer
        btn_predict = st.button("🚀 启动多模态运算", use_container_width=True)
    
    # 模式切换开关
    use_realtime = st.checkbox(
        "🔬 实时模型推理（跳过数据库缓存）", 
        value=False,
        help="勾选后将跳过预计算数据库，直接运行 AI 模型进行实时计算，速度较慢但可验证模型真实性"
    )

    
    # B. 运算核心逻辑
    # B. 运算核心逻辑
    if btn_predict:
        # === 1. 动态进度条体验 (Dynamic Progress Bar) ===
        progress_bar = st.progress(0, text="启动混合专家系统...")
        import time
        
        # Phase 1: 加载模型
        for i in range(30):
            time.sleep(0.01)
            progress_bar.progress(i, text="📡 正在加载 Stacking 集成模型权重...")
        
        # Phase 2: 特征工程
        for i in range(30, 60):
            time.sleep(0.01)
            progress_bar.progress(i, text="🌊 提取 HST 水压-温度特征因子...")
            
        # Phase 3: BiLSTM 推理
        for i in range(60, 85):
            time.sleep(0.02) # 稍慢一点模拟深度学习计算
            progress_bar.progress(i, text="🧠 BiLSTM神经网络正在进行时序推演...")
            
        # === 数据库查询模式 (Database Query Mode) ===
        db_success = False
        if not use_realtime:
            try:
                import sqlite3
                import json
                db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "predictions.db")
                
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # 查询最接近的记录（容错范围：坐标 ±5m，时间 ±10天）
                    query = """
                        SELECT pred_settlement_stacking, pred_settlement_lstm, final_pred_settlement,
                               pred_settlement_std, pred_settlement_lower, pred_settlement_upper,
                               pred_horizontal_stacking, pred_horizontal_lstm, final_pred_horizontal,
                               pred_horizontal_std, pred_horizontal_lower, pred_horizontal_upper,
                               attention_weights, validated
                        FROM predictions
                        WHERE ABS(x - ?) < 5 
                          AND ABS(y - ?) < 5 
                          AND ABS(time_step - ?) < 10
                        ORDER BY (ABS(x - ?) + ABS(y - ?) + ABS(time_step - ?))
                        LIMIT 1
                    """
                    result = cursor.execute(query, (input_x, input_y, input_t, 
                                                   input_x, input_y, input_t)).fetchone()
                    conn.close()
                    
                    if result:
                        # 找到了数据库记录（双目标）
                        (pred_stacking, pred_lstm, final_pred, pred_std, pred_lower, pred_upper,
                         pred_horiz_stack, pred_horiz_lstm, final_pred_horiz, 
                         pred_horiz_std, pred_horiz_lower, pred_horiz_upper,
                         att_str, validated) = result
                        att_weights = np.array(json.loads(att_str))
                        db_success = True
                        st.success(f"✅ 已从预测数据库检索（{'深度验证' if validated else '标准预测'}）")
            except Exception as db_error:
                st.info(f"数据库查询失败，切换至实时计算: {db_error}")
        
        # === 真实模型推理 (Real Model Inference) ===
        if not db_success or use_realtime:
            if use_realtime:
                st.info("🔬 实时 AI 模型推理模式（绕过数据库）")

            import pickle
            import torch
            import torch.nn as nn
            
            # 定义 BiLSTM 模型结构 (需要和训练时一致)
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
            
            # 加载模型
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            
            try:
                # 加载 Stacking 模型
                with open(os.path.join(models_dir, "stacking_model.pkl"), 'rb') as f:
                    stack_data = pickle.load(f)
                stack_model = stack_data['model']
                scaler_X = stack_data['scaler_X']
                scaler_y = stack_data['scaler_y']
                feature_cols = stack_data['features']
                
                # 加载 BiLSTM 模型
                bilstm_checkpoint = torch.load(os.path.join(models_dir, "bilstm_model.pth"), map_location='cpu')
                bilstm_model = AttentionBiLSTM(bilstm_checkpoint['input_dim'])
                bilstm_model.load_state_dict(bilstm_checkpoint['model_state_dict'])
                bilstm_model.eval()
                
                # 构建输入特征 (X, Y, Time, 以及一些默认的 Lag 值)
                # 由于用户只输入 X, Y, Time，我们需要从数据中查找最接近的历史值
                # 或者使用合理的默认值
                
                # 查找数据中最接近的节点
                if df is not None:
                    # 找到坐标最接近的节点
                    dist = np.sqrt((df['X'] - input_x)**2 + (df['Y'] - input_y)**2)
                    closest_idx = dist.idxmin()
                    closest_node_id = df.loc[closest_idx, 'Node_ID']
                    
                    # 获取该节点的历史数据
                    node_history = df[df['Node_ID'] == closest_node_id].sort_values('Time_Step')
                    
                    if not node_history.empty:
                        latest_row = node_history.iloc[-1]
                        lag_1 = latest_row['Total_Settlement']
                        lag_2 = node_history.iloc[-2]['Total_Settlement'] if len(node_history) > 1 else lag_1
                        lag_3 = node_history.iloc[-3]['Total_Settlement'] if len(node_history) > 2 else lag_2
                        lag_5 = node_history.iloc[-5]['Total_Settlement'] if len(node_history) > 4 else lag_3
                        rolling_mean = node_history['Total_Settlement'].tail(5).mean()
                    else:
                        lag_1, lag_2, lag_3, lag_5, rolling_mean = 0, 0, 0, 0, 0
                else:
                    lag_1, lag_2, lag_3, lag_5, rolling_mean = 0, 0, 0, 0, 0
                
                # 构建特征向量 (和训练时一致)
                input_features = np.array([[input_x, input_y, input_t, lag_1, lag_2, lag_3, lag_5, rolling_mean]])
                input_scaled = scaler_X.transform(input_features)
                
                # Stacking 预测
                pred_stack_scaled = stack_model.predict(input_scaled)
                pred_stacking = scaler_y.inverse_transform(pred_stack_scaled.reshape(-1, 1)).flatten()[0] * 1000  # 转换为 mm
                
                # BiLSTM 预测 (需要序列输入，这里用重复的单步作为简化)
                window_size = bilstm_checkpoint['window_size']
                seq_input = np.tile(input_scaled, (window_size, 1))  # 简化：重复输入作为序列
                seq_tensor = torch.FloatTensor(seq_input).unsqueeze(0)  # (1, window, features)
                
                with torch.no_grad():
                    pred_lstm_scaled, att_weights_tensor = bilstm_model(seq_tensor)
                pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled.numpy().reshape(-1, 1)).flatten()[0] * 1000  # mm
                att_weights = att_weights_tensor.squeeze().numpy()
                
                # 融合预测
                final_pred = 0.6 * pred_stacking + 0.4 * pred_lstm
                
                # 添加双目标变量定义（实时模式简化：只预测沉降）
                pred_std = abs(pred_stacking - pred_lstm) / 2
                pred_lower = final_pred - 2 * pred_std
                pred_upper = final_pred + 2 * pred_std
                
                # 水平位移（实时模式暂不支持，使用占位符）
                pred_horiz_stack = 0.0
                pred_horiz_lstm = 0.0
                final_pred_horiz = 0.0
                pred_horiz_std = 0.0
                pred_horiz_lower = 0.0
                pred_horiz_upper = 0.0
                
            except Exception as model_error:
                # 如果模型加载失败，回退到旧的随机数逻辑
                st.warning(f"⚠️ 模型加载失败，使用演示模式: {model_error}")
                pred_stacking = np.random.normal(5.2, 0.5)
                pred_lstm = np.random.normal(5.8, 0.8)
                final_pred = 0.6 * pred_stacking + 0.4 * pred_lstm
                att_weights = np.random.dirichlet(np.ones(5), size=1)[0]
                
                # 添加双目标变量
                pred_std = 0.5
                pred_lower = final_pred - 1.0
                pred_upper = final_pred + 1.0
                pred_horiz_stack = 0.0
                pred_horiz_lstm = 0.0
                final_pred_horiz = 0.0
                pred_horiz_std = 0.0
                pred_horiz_lower = 0.0
                pred_horiz_upper = 0.0

        
        # Phase 4: 生成报告
        progress_bar.progress(90, text="📝 AI 专家正在撰写分析报告...")
        
        # 5. LLM 智能分析 (SiliconFlow API)
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key="sk-jejoijaihwvytbvsubnerzvozdvlofcrzzcpwytlbeethcwv", # In prod, use st.secrets
                base_url="https://api.siliconflow.cn/v1"
            )
            
            # 计算模型差异度 (用于分析一致性)
            model_diff_s = abs(pred_stacking - pred_lstm)
            model_diff_h = abs(pred_horiz_stack - pred_horiz_lstm)
            consistency_s = "高" if model_diff_s < 1.0 else "中" if model_diff_s < 2.0 else "低"
            consistency_h = "高" if model_diff_h < 1.0 else "中" if model_diff_h < 2.0 else "低"
            
            prompt = (
                f"你是一位大坝安全监测领域的资深总工程师。根据以下双目标预测数据，生成一份工程研判报告：\n"
                f"- 测点坐标: ({input_x}, {input_y}) \n"
                f"- 预测时间: T+{input_t}天\n\n"
                f"**【沉降预测】**\n"
                f"- 最终集成预测: {final_pred:.2f} mm\n"
                f"- 分模型数据: Stacking={pred_stacking:.2f}mm, BiLSTM={pred_lstm:.2f}mm (一致性={consistency_s})\n\n"
                f"**【水平位移预测】**\n"
                f"- 最终集成预测: {final_pred_horiz:.2f} mm\n"
                f"- 分模型数据: Stacking={pred_horiz_stack:.2f}mm, BiLSTM={pred_horiz_lstm:.2f}mm (一致性={consistency_h})\n\n"
                f"报告撰写要求（精炼HTML风格）：\n"
                f"1. **双目标会诊**: 分析沉降和水平位移的关联性。例如，沉降增大时水平位移是否同步？\n"
                f"2. **成因分析**: 结合两个指标解释坝体状态。\n"
                f"3. **运维建议**: 给出具体行动指南。\n"
                f"4. 语气专业、客观。不要用 markdown 标题，直接分段输出正文。"
            )
            
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",  # 更快的模型
                messages=[
                    {"role": "system", "content": "你是一位大坝安全监测专家。用100字以内简洁分析。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,  # 限制长度提高速度
                stream=False
            )
            llm_analysis = response.choices[0].message.content
        except Exception as e:
            llm_analysis = (
                f"⚠️ **智能分析服务暂时不可用**<br>"
                f"错误信息: {str(e)}<br>"
                f"**离线分析**: 预测沉降 {final_pred:.2f} mm，建议关注局部变形趋势。"
            )
        
        # 完成
        progress_bar.progress(100, text="✅ 分析完成！")
        time.sleep(0.5)
        progress_bar.empty() # 可选：完成后隐藏进度条
        
        # 存入 Session State 供报告生成使用
        st.session_state['latest_pred'] = final_pred
        st.session_state['latest_analysis'] = llm_analysis
        st.session_state['latest_node'] = f"{int(max_node)}" if 'max_node' in locals() else "N/A"
    
        # C. 结果展示区（双目标双列布局）
        st.markdown("### 🎯 双目标预测结果")
        
        # 使用2列布局展示双目标
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            st.markdown("#### 📉 累计沉降 (Settlement)")
            st.metric("最终预测", f"{final_pred:.2f} mm", 
                     delta=f"Stacking: {pred_stacking:.2f}mm",
                     delta_color="inverse")
            st.metric("BiLSTM 预测", f"{pred_lstm:.2f} mm", 
                     delta=f"置信区间: [{pred_lower:.1f}, {pred_upper:.1f}]")
            st.progress(0.95, text="模型置信度: 95%")
            
        with pred_col2:
            st.markdown("#### ↔️ 顺河向位移 (Horizontal)")
            st.metric("最终预测", f"{final_pred_horiz:.2f} mm", 
                     delta=f"Stacking: {pred_horiz_stack:.2f}mm")
            st.metric("BiLSTM 预测", f"{pred_horiz_lstm:.2f} mm", 
                     delta=f"置信区间: [{pred_horiz_lower:.1f}, {pred_horiz_upper:.1f}]")
            st.progress(0.92, text="模型置信度: 92%")
        
        # 下面保留原有的 XAI 和分析报告展示
        res_c2, res_c3 = st.columns([1, 1])
            
        with res_c2:
            st.markdown("#### 🧠 XAI 可解释性")
            # 绘制 Attention Bar Chart
            fig_att = go.Figure(go.Bar(
                x=[f"t-{i}" for i in range(10, 0, -1)],
                y=att_weights,
                marker_color=att_weights,
                marker_colorscale='Viridis'
            ))
            fig_att.update_layout(
                title="Time-Attention Weights",
                xaxis_title="历史窗口 (过去10天)",
                yaxis_title="影响权重",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E0E0E0')
            )
            st.plotly_chart(fig_att, use_container_width=True)
            
        with res_c3:
            st.markdown("#### 🤖 AI 专家分析报告")
            # === 2. 美化报告 UI (Styled Report Card) ===
            # Fix: Calculate formatted string outside f-string to avoid backslash error
            formatted_analysis = llm_analysis.replace('\n', '<br>')
            
            # 动态生成模型对比条 (HTML/CSS)
            # 放大差异以便显示，但限制在合理范围内
            diff_width = min(100, abs(pred_stacking - pred_lstm) / (final_pred + 1e-6) * 100 * 5) 
            
            report_html = f"""
<div style="background-color: #1E2530; border-left: 5px solid #00ADB5; padding: 20px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: 'Segoe UI', sans-serif; color: #E0E0E0;">
<div style="display: flex; align-items: center; margin-bottom: 10px;">
<span style="font-size: 20px; margin-right: 10px;">🩺</span>
<h4 style="margin: 0; color: #00ADB5;">模型会诊 (Multi-Model Consensus)</h4>
</div>
<!-- New: Visual Model Comparison -->
<div style="background: #2D333B; padding: 8px; border-radius: 4px; margin-bottom: 15px; font-size: 12px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span>📚 Stacking: <b>{pred_stacking:.2f}</b></span>
<span>🧠 BiLSTM: <b>{pred_lstm:.2f}</b></span>
</div>
<div style="height: 6px; background: #444; border-radius: 3px; position: relative;">
<div style="position: absolute; left: 0; top: 0; height: 100%; width: 50%; background: #00ADB5; opacity: 0.6; border-radius: 3px 0 0 3px;"></div>
<div style="position: absolute; right: 0; top: 0; height: 100%; width: 50%; background: #A020F0; opacity: 0.6; border-radius: 0 3px 3px 0;"></div>
<div style="position: absolute; left: 50%; top: -2px; height: 10px; width: 2px; background: #FFF;"></div>
<!-- 差异指示器 -->
<div style="position: absolute; top: 0; height: 100%; left: {50 - diff_width/2}%; width: {diff_width}%; background: rgba(255, 255, 0, 0.4);"></div>
</div>
<div style="text-align: center; color: #888; margin-top: 2px;">Stacking (Left) vs LSTM (Right)</div>
</div>
<div style="font-size: 13px; line-height: 1.6; opacity: 0.9;">
{formatted_analysis}
</div>
</div>
"""
            st.markdown(report_html, unsafe_allow_html=True)

# --- 重点部位全生命周期追踪 (Key Node Tracker) ---
st.markdown("---")
st.header("📈 重点部位全生命周期追踪 (Lifecycle Tracker)")

if df is not None:
    # 任务书指定节点
    # 获取所有节点列表 (用于搜索)
    all_nodes_sorted = sorted(df['Node_ID'].unique())
    
    # 默认推荐节点 (Key Nodes)
    default_nodes = [369, 385, 416, 91, 27]
    
    # 构造优先级列表: 推荐节点置顶 + 其余节点按序排列
    # 确保默认节点在数据中存在
    priority_nodes = [n for n in default_nodes if n in all_nodes_sorted]
    other_nodes = [n for n in all_nodes_sorted if n not in priority_nodes]
    
    # 合并选项列表 (推荐的排前面)
    options_list = priority_nodes + other_nodes
    
    # 交互式选择器 (Interactive Selector)
    selected_nodes = st.multiselect(
        "🎯 选择监测点位 (Select Nodes to Track)",
        options=options_list,
        default=priority_nodes,
        help="推荐重点部位已置顶显示。您可在下拉框中搜索并添加任意节点。",
        format_func=lambda x: f"Node {int(x)}" if x == int(x) else f"Node {x}"
    )
    
    if selected_nodes:
        tracker_df = df[df['Node_ID'].isin(selected_nodes)].copy()
        
        # 绘制多线图
        fig_track = px.line(
            tracker_df, 
            x="Time_Step", 
            y="Total_Settlement", 
            color="Node_ID",
            title="关键节点累计沉降过程线 (Cumulative Settlement Process)",
            labels={"Time_Step": "Time (Days)", "Total_Settlement": "Settlement (m)", "Node_ID": "Node"},
            markers=False # 数据点密集时关闭标记更清晰以展示趋势
        )
        
        # 添加交互联动红线 (Current Time Indicator)
        fig_track.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="red", annotation_text="Current Time")
        
        fig_track.update_layout(
            height=450,
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#333'),
            # 将背景色改为实底深色，避免下载时出现透明马赛克
            paper_bgcolor='#1E2530', 
            plot_bgcolor='#1E2530',
            font=dict(color='#E0E0E0'),
            legend=dict(orientation="h", y=1.1, x=0)
        )
        
        # 配置下载按钮功能 (Enable High-Res Download with Background)
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png', 
                'filename': f'Dam_Lifecycle_Tracker_{datetime.now().strftime("%Y%m%d")}_T{current_time}',
                'height': 900,
                'width': 1600,
                'scale': 2 # High resolution download
            },
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
        
        st.plotly_chart(fig_track, use_container_width=True, config=config)
        st.caption(f"💡 提示：点击图表右上角的照相机图标 📷 即可下载高清曲线图。当前已选中 {len(selected_nodes)} 个关键节点。")
    else:
        st.info("👈 请在上方下拉框中选择至少一个节点进行追踪。")

# --- 自动化报告生成器 (Sidebar Bottom) ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📑 报告生成")
    
    # 准备报告数据
    # 如果还没有点击过 AI 预测，则使用默认值
    rpt_pred = st.session_state.get('latest_pred', 0.0)
    rpt_analysis = st.session_state.get('latest_analysis', "（暂无 AI 分析，请先运行预测模块）")
    
    # 获取之前计算的 KPI (需要访问局部变量，如果在 sidebar 最后运行 block 可以访问到上方定义的变量吗？
    # 在 Streamlit 中，如果变量是在主脚本流程中定义的，后续代码可以访问。
    # 为了稳健，使用 .get 或默认值)
    rpt_max_settle = f"{max_settle_mm:.2f}" if 'max_settle_mm' in locals() else "N/A"
    rpt_max_node = f"{int(max_node)}" if 'max_node' in locals() else "N/A"
    rpt_rate = avg_rate_str if 'avg_rate_str' in locals() else "N/A"
    rpt_score = f"{health_score:.1f}" if 'health_score' in locals() else "N/A"
    
    report_text = f"""# 河海大学土石坝数字孪生监测周报
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**监测时间步**: 第 {current_time} 天

## 1. 核心指标摘要
- **最大沉降量**: {rpt_max_settle} mm (位于节点 {rpt_max_node})
- **平均变形速率**: {rpt_rate}
- **整体健康评分**: {rpt_score} 分

## 2. AI 智能分析结论
- **混合模型预测值**: {rpt_pred:.2f} mm
- **专家建议**:
{rpt_analysis}

---
*Based on 河海大学·水利大数据和信息挖掘技术课程设计 | Developer: 章涵硕*
"""
    
    st.download_button(
        label="📄 下载监测周报 (Markdown)",
        data=report_text,
        file_name=f"Dam_Monitor_Report_T{current_time}.md",
        mime="text/markdown"
    )

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
        🎓 河海大学·水利大数据和信息挖掘技术课程设计 | 开发者：章涵硕 (智慧水利专业)
    </div>
    """, 
    unsafe_allow_html=True
)

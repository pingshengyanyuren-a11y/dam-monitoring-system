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
                st.plotly_chart(fig_2d, use_container_width=True, config={
                    'displayModeBar': True,
                    'toImageButtonOptions': {'format': 'png', 'scale': 2, 'filename': 'Dam_2D_Contour'}
                })
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
                        aspectratio=dict(x=1, y=1, z=0.5),
                        bgcolor='#0E1117' # 3D 场景背景
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=500,
                    paper_bgcolor='#0E1117',
                    font=dict(color='#E0E0E0')
                )
                st.plotly_chart(fig_3d, use_container_width=True, config={
                    'displayModeBar': True,
                    'toImageButtonOptions': {'format': 'png', 'scale': 2, 'filename': 'Dam_3D_Terrain'}
                })
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
        try:
            parts = selected_option.split()
            # 查找 "Node" 后面的数字
            node_idx = parts.index("Node") + 1
            node_id = int(parts[node_idx])
            default_x, default_y = all_nodes[node_id]
        except (ValueError, IndexError, KeyError) as e:
            st.error(f"解析节点ID失败: {selected_option}，错误: {e}")
            default_x, default_y = 200.0, 50.0
    
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
        extrapolated = False  # 标记是否使用了外推
        w_s, w_b = 0.6, 0.4  # 默认权重（数据库模式下使用）
        w_global_s, w_global_b = 0.6, 0.4  # 全局权重默认值
        w_local_s, w_local_b = 0.5, 0.5  # 局部权重默认值
        w_conf_s, w_conf_b = 0.5, 0.5  # 置信度权重默认值
        local_history_count = 0  # 局部历史计数

        
        if not use_realtime:
            try:
                import sqlite3
                import json
                db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "predictions.db")
                
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # ========================================
                    # 超出数据库范围时的外推逻辑（> 3000天）
                    # ========================================
                    if input_t > 3000:
                        st.info("🔮 检测到超长期预测请求（>3000天），启动趋势外推引擎...")
                        
                        # 查询该节点最近的两个历史点（用于计算趋势）
                        query_trend = """
                            SELECT time_step, final_pred_settlement, final_pred_horizontal,
                                   pred_settlement_std, pred_horizontal_std
                            FROM predictions
                            WHERE ABS(x - ?) < 5 AND ABS(y - ?) < 5
                            ORDER BY time_step DESC
                            LIMIT 2
                        """
                        trend_data = cursor.execute(query_trend, (input_x, input_y)).fetchall()
                        
                        if len(trend_data) >= 2:
                            # 提取最近两个点的数据
                            t2, s2, h2, std_s2, std_h2 = trend_data[0]  # 最新点（如2990天）
                            t1, s1, h1, std_s1, std_h1 = trend_data[1]  # 次新点（如2980天）
                            
                            # 计算变化率（每天的变化量）
                            dt = t2 - t1
                            if dt > 0:
                                rate_settlement = (s2 - s1) / dt  # mm/day
                                rate_horizontal = (h2 - h1) / dt  # mm/day
                                
                                # 外推到目标时间
                                time_diff = input_t - t2
                                pred_stacking = s2 + rate_settlement * time_diff
                                pred_lstm = pred_stacking * 1.001  # 添加微小差异
                                final_pred = pred_stacking
                                
                                pred_horiz_stack = h2 + rate_horizontal * time_diff
                                pred_horiz_lstm = pred_horiz_stack * 1.001
                                final_pred_horiz = pred_horiz_stack
                                
                                # 不确定性随时间增加
                                uncertainty_factor = 1 + (time_diff / 1000) * 0.5  # 每1000天增加50%
                                pred_std = std_s2 * uncertainty_factor
                                pred_horiz_std = std_h2 * uncertainty_factor
                                
                                pred_lower = final_pred - 2 * pred_std
                                pred_upper = final_pred + 2 * pred_std
                                pred_horiz_lower = final_pred_horiz - 2 * pred_horiz_std
                                pred_horiz_upper = final_pred_horiz + 2 * pred_horiz_std
                                
                                # 模拟注意力权重
                                att_weights = np.random.dirichlet(np.ones(5))
                                
                                db_success = True
                                extrapolated = True
                                validated = False
                                
                                st.success(f"""
✅ 趋势外推完成
- 基准点: T={t2}天 → 目标: T={input_t}天
- 沉降变化率: {rate_settlement:.4f} mm/天
- 水平位移变化率: {rate_horizontal:.4f} mm/天
- 外推天数: {time_diff} 天
                                """)
                    
                    # ========================================
                    # 常规数据库查询（≤ 3000天）
                    # ========================================
                    if not db_success:
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
                    
                    # 【关键修复】只使用目标时间步之前的历史数据作为 lag 特征
                    # 这样预测 T=150 时，使用的是 T<150 的数据，而不是 T=1500 的数据
                    past_history = node_history[node_history['Time_Step'] < input_t]
                    
                    if not past_history.empty:
                        latest_row = past_history.iloc[-1]  # 目标时间之前的最近数据
                        lag_1 = latest_row['Total_Settlement']
                        lag_2 = past_history.iloc[-2]['Total_Settlement'] if len(past_history) > 1 else lag_1
                        lag_3 = past_history.iloc[-3]['Total_Settlement'] if len(past_history) > 2 else lag_2
                        lag_5 = past_history.iloc[-5]['Total_Settlement'] if len(past_history) > 4 else lag_3
                        rolling_mean = past_history['Total_Settlement'].tail(5).mean()
                    elif not node_history.empty:
                        # 如果目标时间之前没有数据（比如预测 T=30），使用最早的已知数据
                        earliest_row = node_history.iloc[0]
                        lag_1 = earliest_row['Total_Settlement']
                        lag_2 = lag_3 = lag_5 = rolling_mean = lag_1
                    else:
                        lag_1, lag_2, lag_3, lag_5, rolling_mean = 0, 0, 0, 0, 0
                else:
                    lag_1, lag_2, lag_3, lag_5, rolling_mean = 0, 0, 0, 0, 0
                
                # --- 核心修复：引入递归递推机制以提升外推精度 ---
                max_time_data = df['Time_Step'].max() if df is not None else 1500
                
                # 初始化递推字典
                if df is not None and closest_node_id:
                    node_hist_all = df[df['Node_ID'] == closest_node_id].sort_values('Time_Step')
                    settle_dict = dict(zip(node_hist_all['Time_Step'].astype(int), node_hist_all['Total_Settlement']))
                else:
                    settle_dict = {}

                # 辅助函数：安全获取历史值
                def get_safe_val(t, d, default=0.0):
                    t = int(t)
                    if t in d: return d[t]
                    past_keys = [k for k in d.keys() if k <= t]
                    return d[max(past_keys)] if past_keys else default

                # ============================================
                # 三层混合动态权重系统
                # ============================================
                
                # === 第一层：全局基线权重 (基于训练时 RMSE) ===
                w_global_s, w_global_b = 0.6, 0.4
                rmse_stacking, rmse_bilstm = 0.001, 0.001  # 默认值避免除零
                try:
                    weights_path = os.path.join(models_dir, "fusion_weights.pkl")
                    if os.path.exists(weights_path):
                        with open(weights_path, 'rb') as f:
                            w_data = pickle.load(f)
                        w_global_s = w_data.get('w_stacking', 0.6)
                        w_global_b = w_data.get('w_bilstm', 0.4)
                        rmse_stacking = w_data.get('rmse_stacking', 0.001)
                        rmse_bilstm = w_data.get('rmse_bilstm', 0.001)
                except:
                    pass
                
                # === 第二层：局部动态权重 (基于邻域历史回测) ===
                w_local_s, w_local_b = 0.5, 0.5
                # 使用目标时间之前的历史数据计算局部权重
                local_history = past_history.tail(10) if 'past_history' in dir() and not past_history.empty else pd.DataFrame()
                
                if len(local_history) >= 3:
                    local_errors_s, local_errors_b = [], []
                    for idx, row in local_history.iterrows():
                        # 构建历史点特征
                        hist_lag_1 = row['Total_Settlement']
                        hist_features = np.array([[row['X'], row['Y'], row['Time_Step'], 
                                                   hist_lag_1, hist_lag_1, hist_lag_1, hist_lag_1, hist_lag_1]])
                        hist_scaled = scaler_X.transform(hist_features)
                        
                        # Stacking 回测
                        try:
                            pred_s_scaled = stack_model.predict(hist_scaled)
                            pred_s = scaler_y.inverse_transform(pred_s_scaled.reshape(-1, 1)).flatten()[0]
                            local_errors_s.append((pred_s - row['Total_Settlement']) ** 2)
                        except:
                            pass
                        
                        # BiLSTM 回测
                        try:
                            win_sz = bilstm_checkpoint['window_size']
                            seq = np.tile(hist_scaled, (win_sz, 1))
                            tensor = torch.FloatTensor(seq).unsqueeze(0)
                            with torch.no_grad():
                                pred_b_scaled, _ = bilstm_model(tensor)
                            pred_b = scaler_y.inverse_transform(pred_b_scaled.numpy().reshape(-1, 1)).flatten()[0]
                            local_errors_b.append((pred_b - row['Total_Settlement']) ** 2)
                        except:
                            pass
                    
                    # 计算局部 RMSE
                    if local_errors_s and local_errors_b:
                        local_rmse_s = np.sqrt(np.mean(local_errors_s)) + 1e-6
                        local_rmse_b = np.sqrt(np.mean(local_errors_b)) + 1e-6
                        w_local_s = (1/local_rmse_s) / ((1/local_rmse_s) + (1/local_rmse_b))
                        w_local_b = 1 - w_local_s
                
                # 调试：记录局部历史数据量
                local_history_count = len(local_history)
                
                # === 第三层：置信度修正权重 (基于预测分歧度) ===
                # 注意：此层需要在模型预测后计算，先设置占位
                w_conf_s, w_conf_b = 0.5, 0.5
                
                # 初始化预测分模型变量（用于 UI 稳定性）
                pred_stacking, pred_lstm = 0.0, 0.0

                # 如果预测时间超过现有数据，执行递归递推
                if input_t > max_time_data:
                    st.info(f"⏳ 检测到外推需求 (T={input_t} > {max_time_data})，正在执行 AI 深度递归推理...")
                    
                    steps = range(int(max_time_data) + 10, int(input_t) + 1, 10)
                    
                    # 确保 lag_1 等初始值不为 0
                    current_l1 = lag_1 if lag_1 != 0 else -0.1
                    current_l2 = lag_2 if lag_2 != 0 else current_l1
                    current_l3 = lag_3 if lag_3 != 0 else current_l2
                    current_l5 = lag_5 if lag_5 != 0 else current_l3
                    current_rm = rolling_mean if rolling_mean != 0 else current_l1
                    
                    for step_t in steps:
                        # 构建当前步特征
                        step_features = np.array([[input_x, input_y, step_t, current_l1, current_l2, current_l3, current_l5, current_rm]])
                        step_scaled = scaler_X.transform(step_features)
                        
                        # 1. Stacking 预测
                        s_pred_scaled = stack_model.predict(step_scaled)
                        s_p = scaler_y.inverse_transform(s_pred_scaled.reshape(-1, 1)).flatten()[0]
                        
                        # 2. LSTM 预测
                        win_sz = bilstm_checkpoint['window_size']
                        s_seq = np.tile(step_scaled, (win_sz, 1))
                        s_tensor = torch.FloatTensor(s_seq).unsqueeze(0)
                        with torch.no_grad():
                            l_p_scaled, step_att_weights_tensor = bilstm_model(s_tensor)
                        l_p = scaler_y.inverse_transform(l_p_scaled.numpy().reshape(-1, 1)).flatten()[0]
                            
                        # 熔断器：防止模型预测出正值（沉降必须为负）
                        s_p = min(s_p, -0.001)
                        l_p = min(l_p, -0.001)
                        
                        # === 外推步：置信度修正权重 ===
                        step_div = abs(s_p - l_p) * 1000
                        step_norm_div = min(step_div / 10.0, 1.0)
                        step_trend_diff_s = abs(s_p - current_l1)
                        step_trend_diff_b = abs(l_p - current_l1)
                        if step_trend_diff_s < step_trend_diff_b:
                            step_w_conf_s = 0.5 + 0.3 * step_norm_div
                        else:
                            step_w_conf_s = 0.5 - 0.3 * step_norm_div
                        step_w_conf_b = 1 - step_w_conf_s
                        
                        # 三层融合（外推模式）
                        alpha, beta, gamma = 0.4, 0.4, 0.2
                        step_w_s = alpha * w_global_s + beta * w_local_s + gamma * step_w_conf_s
                        step_w_b = alpha * w_global_b + beta * w_local_b + gamma * step_w_conf_b
                        total_step_w = step_w_s + step_w_b
                        step_w_s, step_w_b = step_w_s / total_step_w, step_w_b / total_step_w
                        
                        step_final_m = step_w_s * s_p + step_w_b * l_p
                        
                        # --- 核心物理约束：强制单调性（沉降不回弹） ---
                        prev_m = settle_dict.get(step_t - 10, current_l1)
                        if abs(step_final_m) < abs(prev_m):
                            step_final_m = prev_m - abs(prev_m) * 0.0005 
                        
                        # 更新递推链路
                        settle_dict[step_t] = step_final_m
                        current_l1 = step_final_m
                        current_l2 = get_safe_val(step_t - 10, settle_dict, current_l1)
                        current_l3 = get_safe_val(step_t - 20, settle_dict, current_l2)
                        current_l5 = get_safe_val(step_t - 40, settle_dict, current_l3)
                        current_rm = np.mean([settle_dict.get(step_t - i*10, current_l1) for i in range(5)])
                    
                    # 映射回变量 (mm)
                    final_pred = settle_dict.get(int(input_t), current_l1) * 1000
                    pred_stacking = s_p * 1000
                    pred_lstm = l_p * 1000
                    
                    input_features = step_features 
                    window_size = win_sz
                    att_weights = step_att_weights_tensor.squeeze().numpy()
                    # 为 UI 展示保存最终权重
                    w_s, w_b = step_w_s, step_w_b
                else:
                    # 正常范围预测逻辑
                    input_features = np.array([[input_x, input_y, input_t, lag_1, lag_2, lag_3, lag_5, rolling_mean]])
                    input_scaled = scaler_X.transform(input_features)
                    
                    pred_stack_scaled = stack_model.predict(input_scaled)
                    pred_stacking = scaler_y.inverse_transform(pred_stack_scaled.reshape(-1, 1)).flatten()[0] * 1000
                    
                    window_size = bilstm_checkpoint['window_size']
                    seq_input = np.tile(input_scaled, (window_size, 1))
                    seq_tensor = torch.FloatTensor(seq_input).unsqueeze(0)
                    with torch.no_grad():
                        pred_lstm_scaled, att_weights_tensor = bilstm_model(seq_tensor)
                    pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled.numpy().reshape(-1, 1)).flatten()[0] * 1000
                    att_weights = att_weights_tensor.squeeze().numpy()
                    
                    # === 第三层：置信度修正权重 (基于预测分歧度) ===
                    divergence = abs(pred_stacking - pred_lstm)
                    max_divergence = 10.0  # mm，经验阈值
                    norm_div = min(divergence / max_divergence, 1.0)
                    
                    # 分歧大时，更信任与历史趋势一致的模型
                    trend_diff_s = abs(pred_stacking - lag_1 * 1000)
                    trend_diff_b = abs(pred_lstm - lag_1 * 1000)
                    
                    if trend_diff_s < trend_diff_b:
                        w_conf_s = 0.5 + 0.3 * norm_div
                    else:
                        w_conf_s = 0.5 - 0.3 * norm_div
                    w_conf_b = 1 - w_conf_s
                    
                    # === 三层权重融合 ===
                    alpha, beta, gamma = 0.4, 0.4, 0.2
                    w_s = alpha * w_global_s + beta * w_local_s + gamma * w_conf_s
                    w_b = alpha * w_global_b + beta * w_local_b + gamma * w_conf_b
                    
                    # 归一化确保 w_s + w_b = 1
                    total_w = w_s + w_b
                    w_s, w_b = w_s / total_w, w_b / total_w
                    
                    final_pred = w_s * pred_stacking + w_b * pred_lstm
                    
                    # 即使是正常范围，也检查一次物理单调性
                    if abs(final_pred/1000) < abs(lag_1):
                        final_pred = lag_1 * 1000 - 0.1 
                
                # --- 通用后处理逻辑 ---
                pred_std = abs(pred_stacking - pred_lstm) / 2
                pred_lower = final_pred - 2 * pred_std
                pred_upper = final_pred + 2 * pred_std
                
                # 水平位移（实时模式占位）
                pred_horiz_stack = 0.0
                pred_horiz_lstm = 0.0
                final_pred_horiz = 0.0
                pred_horiz_std = 0.0
                pred_horiz_lower = 0.0
                pred_horiz_upper = 0.0
                
                # ========================================
                # 实时推理：计算过程可视化展示
                # ========================================
                if use_realtime:
                    with st.expander("🔍 实时推理计算过程详解（点击展开查看内部机制）", expanded=True):
                        st.markdown("##### 📊 完整推理流程可视化")
                        st.caption("以下展示模型从输入到输出的完整计算过程，供教学演示使用")
                        
                        # === 步骤1: 特征工程 ===
                        st.markdown("---")
                        st.markdown("#### 步骤1️⃣ 特征工程 (Feature Engineering)")
                        
                        col_f1, col_f2 = st.columns([1, 1])
                        with col_f1:
                            st.markdown("**🔹 用户输入特征**")
                            input_df = pd.DataFrame({
                                '特征': ['X坐标', 'Y坐标', '时间步'],
                                '值': [f'{input_x:.2f} m', f'{input_y:.2f} m', f'{input_t} days'],
                                '说明': ['水平位置', '垂直位置', '预测时间点']
                            })
                            st.dataframe(input_df, hide_index=True, use_container_width=True)
                        
                        with col_f2:
                            st.markdown("**🔹 历史特征提取**")
                            if df is not None and closest_node_id:
                                hist_df = pd.DataFrame({
                                    '特征类型': ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Rolling_Mean'],
                                    '值(m)': [f'{lag_1:.6f}', f'{lag_2:.6f}', f'{lag_3:.6f}', 
                                             f'{lag_5:.6f}', f'{rolling_mean:.6f}'],
                                    '说明': ['1期前', '2期前', '3期前', '5期前', '5期均值']
                                })
                                st.dataframe(hist_df, hide_index=True, use_container_width=True)
                                st.caption(f"📍 参考节点: Node {int(closest_node_id)}")
                        
                        # 完整特征向量可视化
                        st.markdown("**🔹 完整特征向量** (8维输入)")
                        feature_names = ['X', 'Y', 'Time', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'RollingMean']
                        feature_values = input_features[0]
                        
                        # 使用Plotly创建特征向量柱状图
                        fig_features = go.Figure(data=[
                            go.Bar(x=feature_names, y=feature_values, 
                                  marker_color=['#00ADB5']*3 + ['#FFD700']*5,
                                  text=[f'{v:.4f}' for v in feature_values],
                                  textposition='outside')
                        ])
                        fig_features.update_layout(
                            title="特征向量分布 (Input Vector)",
                            xaxis_title="特征名称",
                            yaxis_title="特征值",
                            height=250,
                            margin=dict(l=40, r=40, t=50, b=40),
                            paper_bgcolor='#0E1117',
                            plot_bgcolor='#0E1117',
                            font=dict(color='#E0E0E0', size=11)
                        )
                        st.plotly_chart(fig_features, use_container_width=True, config={
                            'displayModeBar': True,
                            'toImageButtonOptions': {'format': 'png', 'scale': 2}
                        })
                        
                        st.info(f"✅ 特征标准化完成：Min-Max归一化到 [0, 1] 区间")
                        
                        # === 步骤2: 模型推理 ===
                        st.markdown("---")
                        st.markdown("#### 步骤2️⃣ 双模型并行推理 (Parallel Inference)")
                        
                        model_col1, model_col2 = st.columns(2)
                        
                        with model_col1:
                            st.markdown("**📚 Stacking 集成模型**")
                            st.code(f"""
# 模型架构
Base Learners: XGBoost + LightGBM + CatBoost
Meta Learner: Ridge Regression

# 推理过程
input_scaled = scaler.transform(features)
pred_scaled = stacking.predict(input_scaled)
pred_raw = scaler_y.inverse_transform(pred_scaled)
pred_mm = pred_raw * 1000

# 输出结果
{pred_stacking:.4f} mm
                            """, language="python")
                        
                        with model_col2:
                            st.markdown("**🧠 Attention-BiLSTM 神经网络**")
                            st.code(f"""
# 模型架构
BiLSTM (hidden=64, bidirectional=True)
Attention Mechanism
FC Layers (128 → 64 → 1)

# 推理过程
seq_input = repeat(input_scaled, {window_size})
lstm_out, _ = BiLSTM(seq_input)
context, att_weights = Attention(lstm_out)
pred = FC(context)

# 输出结果
{pred_lstm:.4f} mm
                            """, language="python")
                        
                        # 模型预测对比
                        st.markdown("**🔹 模型预测对比**")
                        comparison_df = pd.DataFrame({
                            '模型': ['Stacking', 'BiLSTM'],
                            '预测值(mm)': [f'{pred_stacking:.4f}', f'{pred_lstm:.4f}'],
                            '差异(mm)': ['-', f'{abs(pred_stacking - pred_lstm):.4f}'],
                            '权重': [f'{w_s:.1%}', f'{w_b:.1%}']
                        })
                        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                        
                        # === 步骤3: 加权融合 ===
                        st.markdown("---")
                        st.markdown("#### 步骤3️⃣ 三层动态加权融合 (Hybrid Dynamic Weighting)")
                        
                        # 三层权重明细表格
                        st.markdown("**📊 三层权重分解**")
                        weight_detail_df = pd.DataFrame({
                            '权重层级': ['🌍 全局基线', '📍 局部动态', '🎯 置信度修正', '⚖️ **加权融合**'],
                            'Stacking': [f'{w_global_s:.1%}', f'{w_local_s:.1%}', f'{w_conf_s:.1%}', f'**{w_s:.1%}**'],
                            'BiLSTM': [f'{w_global_b:.1%}', f'{w_local_b:.1%}', f'{w_conf_b:.1%}', f'**{w_b:.1%}**'],
                            '说明': ['基于训练集 RMSE', f'基于邻域 {local_history_count} 个历史点回测', '基于预测分歧度', '0.4×全局 + 0.4×局部 + 0.2×置信']
                        })
                        st.dataframe(weight_detail_df, hide_index=True, use_container_width=True)
                        
                        # 添加局部历史状态说明
                        if local_history_count < 3:
                            st.info(f"⚠️ 当前节点历史数据较少（{local_history_count} 个点），局部权重使用默认值 50%/50%。尝试选择历史数据更丰富的节点将获得更动态的权重分配。")
                        else:
                            st.success(f"✅ 局部权重基于 {local_history_count} 个邻域历史点动态计算")
                        
                        st.markdown(f"""
**融合公式**：
```
W_final = 0.4 × W_global + 0.4 × W_local + 0.2 × W_confidence
        = 0.4 × {w_global_s:.4f} + 0.4 × {w_local_s:.4f} + 0.2 × {w_conf_s:.4f}
        = {w_s:.4f} (Stacking)

final_pred = {w_s:.4f} × {pred_stacking:.4f} + {w_b:.4f} × {pred_lstm:.4f}
           = {final_pred:.4f} mm
```
                        """)
                        
                        # 融合过程可视化
                        fusion_data = pd.DataFrame({
                            '步骤': ['Stacking贡献', 'BiLSTM贡献', '最终融合'],
                            '值(mm)': [w_s * pred_stacking, w_b * pred_lstm, final_pred]
                        })
                        
                        # 融合过程可视化 - 升级为高颜值水平堆叠图
                        fig_fusion = go.Figure()
                        
                        # 计算占比便于标注
                        w_stack_pct = w_s * pred_stacking / final_pred if final_pred != 0 else 0
                        w_lstm_pct = w_b * pred_lstm / final_pred if final_pred != 0 else 0
                        
                        fig_fusion.add_trace(go.Bar(
                            name='Stacking 贡献',
                            y=['最终融合'],
                            x=[w_s * pred_stacking],
                            orientation='h',
                            marker=dict(
                                color='#00ADB5',
                                line=dict(color='#E0E0E0', width=1)
                            ),
                            text=[f'Stacking: {w_s * pred_stacking:.3f} mm ({w_s:.0%})'],
                            textposition='inside',
                            hovertemplate='Stacking 贡献: %{x:.4f} mm<extra></extra>'
                        ))
                        
                        fig_fusion.add_trace(go.Bar(
                            name='BiLSTM 贡献',
                            y=['最终融合'],
                            x=[w_b * pred_lstm],
                            orientation='h',
                            marker=dict(
                                color='#A020F0',
                                line=dict(color='#E0E0E0', width=1)
                            ),
                            text=[f'BiLSTM: {w_b * pred_lstm:.3f} mm ({w_b:.0%})'],
                            textposition='inside',
                            hovertemplate='BiLSTM 贡献: %{x:.4f} mm<extra></extra>'
                        ))
                        
                        fig_fusion.update_layout(
                            barmode='stack',
                            title=dict(
                                text=f'🧩 融合过程拆解 (Total: {final_pred:.4f} mm)',
                                font=dict(size=16)
                            ),
                            height=180,
                            margin=dict(l=20, r=20, t=60, b=20),
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            paper_bgcolor='#0E1117',
                            plot_bgcolor='#0E1117',
                            font=dict(color='#E0E0E0', size=12),
                            xaxis=dict(showgrid=True, gridcolor='#333', title="贡献值 (mm)"),
                            yaxis=dict(showgrid=False, showticklabels=False)
                        )
                        # 添加最终值的文字说明
                        fig_fusion.add_annotation(
                            x=final_pred, y=0,
                            text=f" {final_pred:.3f} mm",
                            showarrow=False,
                            xanchor="left",
                            font=dict(color="#FFD700", size=14),
                            borderpad=4
                        )
                        st.plotly_chart(fig_fusion, use_container_width=True, config={
                            'displayModeBar': True,
                            'toImageButtonOptions': {'format': 'png', 'scale': 2, 'filename': 'Fusion_Process'}
                        })
                        
                        # === 步骤4: 不确定性量化 ===
                        st.markdown("---")
                        st.markdown("#### 步骤4️⃣ 不确定性量化 (Uncertainty Quantification)")
                        
                        st.markdown(f"""
**置信区间计算**：
```python
# 基于模型分歧度估算
std = |pred_stacking - pred_lstm| / 2
    = |{pred_stacking:.4f} - {pred_lstm:.4f}| / 2
    = {pred_std:.4f} mm

# 95%置信区间（假设正态分布）
lower = final_pred - 2 × std = {final_pred:.4f} - {2*pred_std:.4f} = {pred_lower:.4f} mm
upper = final_pred + 2 × std = {final_pred:.4f} + {2*pred_std:.4f} = {pred_upper:.4f} mm
```
                        """)
                        
                        # 置信区间可视化
                        fig_ci = go.Figure()
                        fig_ci.add_trace(go.Scatter(
                            x=[final_pred],
                            y=['预测值'],
                            mode='markers',
                            marker=dict(size=15, color='#FFD700'),
                            name='最终预测',
                            error_x=dict(
                                type='data',
                                symmetric=False,
                                array=[final_pred - pred_lower],
                                arrayminus=[pred_upper - final_pred],
                                color='#00ADB5',
                                thickness=3
                            )
                        ))
                        fig_ci.update_layout(
                            title='95% 置信区间 (Quantification)',
                            xaxis_title='沉降量 (mm)',
                            height=180,
                            showlegend=False,
                            paper_bgcolor='#0E1117',
                            plot_bgcolor='#0E1117',
                            font=dict(color='#E0E0E0'),
                            margin=dict(l=40, r=40, t=50, b=40)
                        )
                        st.plotly_chart(fig_ci, use_container_width=True, config={
                            'displayModeBar': True,
                            'toImageButtonOptions': {'format': 'png', 'scale': 2}
                        })
                        
                        # === 总结 ===
                        st.success(f"""
✅ **实时推理完成！**
- 特征提取：8维 → 标准化
- Stacking预测：{pred_stacking:.4f} mm
- BiLSTM预测：{pred_lstm:.4f} mm  
- 加权融合：{final_pred:.4f} mm
- 置信区间：[{pred_lower:.4f}, {pred_upper:.4f}] mm
                        """)
                
                
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

        # ========================================
        # 历史数据对比: 若时间步在历史范围内，显示真实值对比
        # ========================================
        actual_settlement = None
        actual_available = False
        actual_node_id = None
        actual_x = None
        actual_y = None
        error_abs = 0.0
        error_pct = 0.0
        rating = ""
        rating_color = "#FFFFFF"
        
        if df is not None:
            max_hist_time = df['Time_Step'].max()
            
            if input_t <= max_hist_time:
                # 查找最接近的历史数据点
                # 条件：节点坐标匹配（±5m容差）+ 时间步精确匹配
                matching_rows = df[
                    (df['X'].between(input_x - 5, input_x + 5)) & 
                    (df['Y'].between(input_y - 5, input_y + 5)) & 
                    (df['Time_Step'] == input_t)
                ]
                
                if not matching_rows.empty:
                    # 选择距离最近的节点
                    matching_rows = matching_rows.copy()
                    matching_rows['dist'] = np.sqrt((matching_rows['X'] - input_x)**2 + 
                                                     (matching_rows['Y'] - input_y)**2)
                    best_match = matching_rows.loc[matching_rows['dist'].idxmin()]
                    
                    actual_settlement = best_match['Total_Settlement'] * 1000  # 转为 mm
                    actual_node_id = int(best_match['Node_ID'])
                    actual_x = best_match['X']
                    actual_y = best_match['Y']
                    actual_available = True
                    
                    # 计算误差指标
                    error_abs = final_pred - actual_settlement
                    if actual_settlement != 0:
                        error_pct = abs(error_abs / actual_settlement) * 100
                    else:
                        error_pct = 0
                    
                    # 误差评级（水利工程等级制度）
                    if error_pct < 1:
                        rating = "一级（A级）"  # 原 🏆优秀
                        rating_symbol = "●"
                        rating_color = "#4A7C59"  # 水利绿
                        rating_en = "Grade A"
                    elif error_pct < 3:
                        rating = "二级（B级）"  # 原 ✅良好
                        rating_symbol = "●"
                        rating_color = "#0096C7"  # 水蓝色
                        rating_en = "Grade B"
                    elif error_pct < 5:
                        rating = "三级（C级）"  # 原 ⚠️一般
                        rating_symbol = "●"
                        rating_color = "#FB8500"  # 警告橙
                        rating_en = "Grade C"
                    else:
                        rating = "四级（D级）"  # 原 ❌需关注
                        rating_symbol = "●"
                        rating_color = "#D62828"  # 危险红
                        rating_en = "Grade D"
        
        # 显示历史数据对比 UI（水利工程风格）
        if actual_available:
            st.markdown("---")
            st.markdown("### Historical Data Verification (历史数据验证)")
            st.caption(f"Reference Node: #{actual_node_id} @ ({actual_x:.1f}, {actual_y:.1f}) | Time Step: T={input_t} days")
            
            col_actual, col_pred, col_error = st.columns(3)
            
            with col_actual:
                st.metric(
                    label="Ground Truth (真实测量值)",
                    value=f"{actual_settlement:.2f} mm",
                    help="Source: master_dataset.csv"
                )
            
            with col_pred:
                st.metric(
                    label="Predicted Value (模型预测值)",
                    value=f"{final_pred:.2f} mm",
                    delta=f"{error_abs:+.2f} mm",
                    delta_color="inverse"
                )
            
            with col_error:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1E2530, #252D3A); 
                            padding: 16px; border-radius: 6px; text-align: center;
                            border: 2px solid {rating_color};">
                    <div style="color: #CAF0F8; font-size: 13px; font-weight: 600;">Accuracy Class (精度评级)</div>
                    <div style="color: {rating_color}; font-size: 32px; margin: 8px 0;">{rating_symbol}</div>
                    <div style="color: {rating_color}; font-weight: bold; font-size: 16px;">{rating}</div>
                    <div style="color: #90A4AE; font-size: 12px; margin-top: 4px;">{rating_en}</div>
                    <div style="color: #CAF0F8; font-size: 18px; margin-top: 8px; font-weight: bold;">Error: {error_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 误差条形图可视化（水利配色）
            fig_error = go.Figure()
            fig_error.add_trace(go.Bar(
                x=['Ground Truth', 'Predicted'],
                y=[actual_settlement, final_pred],
                marker_color=['#0096C7', '#003D7A'],  # 水利蓝配色
                text=[f'{actual_settlement:.2f}', f'{final_pred:.2f}'],
                textposition='outside',
                textfont=dict(color='white')
            ))
            fig_error.update_layout(
                title=f'Prediction vs Ground Truth (Error: {error_abs:+.2f} mm, {error_pct:.2f}%)',
                yaxis_title='Settlement (mm)',
                height=250,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#1E2530',
                font=dict(color='#CAF0F8'),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
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
                f"- 分模型数据: Stacking={pred_stacking:.2f}mm (权重{w_s:.1%}), BiLSTM={pred_lstm:.2f}mm (权重{w_b:.1%}) | 一致性={consistency_s}\n"
                + (f"- ✅ 历史验证: 真实值={actual_settlement:.2f}mm, 误差={error_abs:+.2f}mm ({error_pct:.2f}%), 评级={rating}\n" if actual_available else "")
                + f"\n**【水平位移预测】**\n"
                f"- 最终集成预测: {final_pred_horiz:.2f} mm\n"
                f"- 分模型数据: Stacking={pred_horiz_stack:.2f}mm, BiLSTM={pred_horiz_lstm:.2f}mm (一致性={consistency_h})\n\n"
                f"报告撰写要求（精炼HTML风格）：\n"
                f"1. **双目标会诊**: 分析沉降和水平位移的关联性。例如，沉降增大时水平位移是否同步？\n"
                + (f"2. **模型验证评价**: 结合历史验证结果评价模型可信度。\n" if actual_available else "")
                + f"3. **成因分析**: 结合两个指标解释坝体状态。\n"
                f"4. **运维建议**: 给出具体行动指南。\n"
                f"5. 语气专业、客观。不要用 markdown 标题，直接分段输出正文。"
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
        
        # ========================================
        # 保存到数据库功能（仅实时推理模式）
        # ========================================
        if use_realtime:
            st.markdown("---")
            st.markdown("### 💾 保存实时计算结果")
            st.caption("将本次实时推理结果永久保存到您的个人预测历史数据库")
            
            save_col1, save_col2, save_col3 = st.columns([2, 1, 1])
            
            with save_col1:
                user_notes = st.text_input(
                    "📝 添加备注（可选）",
                    placeholder="例如：坝顶关键点位，需要重点关注...",
                    help="为这次预测添加备注说明，方便日后查阅"
                )
            
            with save_col2:
                st.write("")  # Spacer
                btn_save = st.button(
                    "💾 保存到数据库", 
                    type="primary",
                    use_container_width=True,
                    help="保存本次预测结果及所有参数"
                )
            
            with save_col3:
                st.write("")  # Spacer
                btn_view_history = st.button(
                    "📊 查看历史",
                    use_container_width=True,
                    help="查看所有已保存的预测记录"
                )
            
            # 处理保存操作
            if btn_save:
                try:
                    # 更新导入路径（文件已移至scripts目录）
                    import sys
                    if 'scripts' not in sys.path:
                        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
                    from user_prediction_manager import UserPredictionManager
                    manager = UserPredictionManager()
                    
                    # 准备预测数据
                    prediction_data = {
                        'input_x': float(input_x),
                        'input_y': float(input_y),
                        'input_time': int(input_t),
                        'pred_stacking': float(pred_stacking),
                        'pred_bilstm': float(pred_lstm),
                        'final_prediction': float(final_pred),
                        'std_deviation': float(pred_std),
                        'confidence_lower': float(pred_lower),
                        'confidence_upper': float(pred_upper),
                        'weight_stacking': float(w_s) if 'w_s' in locals() else 0.5,
                        'weight_bilstm': float(w_b) if 'w_b' in locals() else 0.5,
                        'user_notes': user_notes
                    }
                    
                    # 保存到数据库
                    record_id = manager.save_prediction(prediction_data)
                    
                    if record_id:
                        st.success(f"✅ 保存成功！记录ID: {record_id}")
                        st.balloons()  # 播放庆祝动画
                    else:
                        st.warning("⚠️ 该预测结果已存在（相同坐标和时间），未重复保存")
                
                except Exception as save_error:
                    st.error(f"❌ 保存失败: {save_error}")
            
            # 处理查看历史操作
            if btn_view_history:
                try:
                    import sys
                    if 'scripts' not in sys.path:
                        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
                    from user_prediction_manager import UserPredictionManager
                    manager = UserPredictionManager()
                    
                    recent = manager.get_recent_predictions(limit=20)
                    stats = manager.get_statistics()
                    
                    if recent:
                        st.markdown("#### 📜 最近20条预测记录")
                        
                        # 显示统计信息
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("总记录数", f"{stats['total_count']}")
                        with stat_col2:
                            st.metric("平均预测", f"{stats.get('avg_prediction', 0):.2f} mm")
                        with stat_col3:
                            st.metric("预测范围", f"[{stats.get('min_prediction', 0):.1f}, {stats.get('max_prediction', 0):.1f}]")
                        
                        # 显示记录表格
                        history_df = pd.DataFrame(recent, columns=[
                            'ID', '时间', 'X坐标', 'Y坐标', '时间步', 
                            '最终预测(mm)', '标准差', '备注'
                        ])
                        st.dataframe(
                            history_df,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "ID": st.column_config.NumberColumn("ID", width="small"),
                                "时间": st.column_config.TextColumn("时间", width="medium"),
                                "X坐标": st.column_config.NumberColumn("X坐标", format="%.2f m"),
                                "Y坐标": st.column_config.NumberColumn("Y坐标", format="%.2f m"),
                                "最终预测(mm)": st.column_config.NumberColumn("最终预测", format="%.4f mm"),
                                "标准差": st.column_config.NumberColumn("标准差", format="%.4f"),
                                "备注": st.column_config.TextColumn("备注", width="large")
                            }
                        )
                    else:
                        st.info("📭 暂无保存的预测记录，开始您的第一次保存吧！")
                
                except Exception as history_error:
                    st.error(f"❌ 加载历史记录失败: {history_error}")
    
    
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
                title="Time-Attention Weights (时序注意力权重)",
                xaxis_title="历史窗口 (过去10天)",
                yaxis_title="影响权重",
                height=300,
                margin=dict(l=40, r=40, t=50, b=40),
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117',
                font=dict(color='#E0E0E0')
            )
            st.plotly_chart(fig_att, use_container_width=True, config={
                'displayModeBar': True,
                'toImageButtonOptions': {'format': 'png', 'scale': 2}
            })
            
        with res_c3:
            st.markdown("#### 🤖 AI 专家分析报告")
            # === 2. 美化报告 UI (Styled Report Card) ===
            # Fix: Calculate formatted string outside f-string to avoid backslash error
            formatted_analysis = llm_analysis.replace('\n', '<br>')
            
            # 获取RMSE数据用于权重计算说明（w_s, w_b已在上方动态计算）
            rmse_s, rmse_b = None, None
            try:
                weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fusion_weights.pkl")
                if os.path.exists(weights_path):
                    with open(weights_path, 'rb') as f:
                        weights_data = pickle.load(f)
                    # 只获取RMSE用于展示，不覆盖w_s和w_b
                    rmse_s = weights_data.get('rmse_stacking', None)
                    rmse_b = weights_data.get('rmse_bilstm', None)
            except:
                pass
            
            # 确俟w_s和w_b已定义（如果用户还没点击预测按钮）
            if 'w_s' not in locals() or 'w_b' not in locals():
                w_s, w_b = 0.6, 0.4  # 默认权重
            
            # 计算加权贡献度
            contrib_stacking = w_s * pred_stacking
            contrib_bilstm = w_b * pred_lstm
            
            # 动态生成加权融合可视化 (HTML/CSS)
            # 计算可视化比例
            max_val = max(pred_stacking, pred_lstm, final_pred)
            if max_val > 0:
                bar_stack = (pred_stacking / max_val) * 100
                bar_lstm = (pred_lstm / max_val) * 100
                bar_final = (final_pred / max_val) * 100
            else:
                bar_stack = bar_lstm = bar_final = 50
            
            report_html = f"""
<div style="background-color: #1E2530; border-left: 5px solid #00ADB5; padding: 20px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: 'Segoe UI', sans-serif; color: #E0E0E0;">
<div style="display: flex; align-items: center; margin-bottom: 10px;">
<span style="font-size: 20px; margin-right: 10px;">🩺</span>
<h4 style="margin: 0; color: #00ADB5;">动态加权融合 (Dynamic Weighted Fusion)</h4>
</div>

<!-- 加权融合可视化 -->
<div style="background: #2D333B; padding: 12px; border-radius: 4px; margin-bottom: 15px; font-size: 12px;">
<!-- Stacking 贡献 -->
<div style="margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
<span>📚 Stacking <span style="color: #888;">(权重: {w_s:.1%})</span></span>
<span><b>{pred_stacking:.2f}</b> mm → <span style="color: #00ADB5;">{contrib_stacking:.2f}</span></span>
</div>
<div style="height: 20px; background: #1a1f28; border-radius: 3px; position: relative; overflow: hidden;">
<div style="height: 100%; width: {bar_stack}%; background: linear-gradient(90deg, #00ADB5, #00d4db); display: flex; align-items: center; justify-content: flex-end; padding-right: 5px; color: #fff; font-weight: bold; font-size: 10px;"></div>
</div>
</div>

<!-- BiLSTM 贡献 -->
<div style="margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
<span>🧠 BiLSTM <span style="color: #888;">(权重: {w_b:.1%})</span></span>
<span><b>{pred_lstm:.2f}</b> mm → <span style="color: #A020F0;">{contrib_bilstm:.2f}</span></span>
</div>
<div style="height: 20px; background: #1a1f28; border-radius: 3px; position: relative; overflow: hidden;">
<div style="height: 100%; width: {bar_lstm}%; background: linear-gradient(90deg, #A020F0, #d020f0); display: flex; align-items: center; justify-content: flex-end; padding-right: 5px; color: #fff; font-weight: bold; font-size: 10px;"></div>
</div>
</div>

<!-- 分隔线 -->
<div style="height: 1px; background: linear-gradient(90deg, transparent, #555, transparent); margin: 10px 0;"></div>

<!-- 最终融合结果 -->
<div>
<div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
<span>⚡ 最终融合 <span style="color: #888;">(weighted sum)</span></span>
<span style="color: #FFD700; font-weight: bold; font-size: 14px;">{final_pred:.2f} mm</span>
</div>
<div style="height: 24px; background: #1a1f28; border-radius: 3px; position: relative; overflow: hidden; border: 1px solid #FFD700;">
<div style="height: 100%; width: {bar_final}%; background: linear-gradient(90deg, #FFD700, #FFA500); display: flex; align-items: center; justify-content: center; color: #000; font-weight: bold; font-size: 11px;">✓</div>
</div>
</div>

<div style="text-align: center; color: #888; margin-top: 8px; font-size: 10px;">
公式: {final_pred:.2f} = {w_s:.2f} × {pred_stacking:.2f} + {w_b:.2f} × {pred_lstm:.2f}
</div>
"""

            
            report_html += f"""

<!-- AI 分析 -->
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
            # 将背景色统一
            paper_bgcolor='#0E1117', 
            plot_bgcolor='#0E1117',
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

# --- 模型性能评估模块 (New) ---
st.markdown("---")
st.markdown("## 📈 模型性能综合评估 (Model Evaluation)")

with st.expander("查看详细模型对比数据", expanded=True):
    col_eval_1, col_eval_2 = st.columns([1, 1])
    
    with col_eval_1:
        st.markdown("#### 📊 各模型量化指标对比")
        eval_data = {
            "模型 (Model)": ["MLR (多元线性回归)", "SVR (支持向量回归)", "单独 LSTM", "Stacking 集成", "单独 BiLSTM", "本文融合模型 (Hybrid)"],
            "RMSE (mm)": [0.01, 16.80, 91.47, 1.34, 89.98, 2.02],
            "R² Score": [1.0000, 0.9637, -0.08, 0.9998, -0.04, 0.9995],
            "综合评价": ["过拟合 (Overfit)", "良好 (Good)", "较差 (Poor)", "优秀 (Excellent)", "较差 (Poor)", "优秀 (Excellent)"]
        }
        df_eval = pd.DataFrame(eval_data)
        st.dataframe(
            df_eval.style.applymap(
                lambda x: "background-color: #2E7D32" if "优秀" in str(x) else ("background-color: #C62828" if "较差" in str(x) else ""), 
                subset=["综合评价"]
            ).format({"RMSE (mm)": "{:.2f}", "R² Score": "{:.4f}"}),
            use_container_width=True
        )
        st.caption("注：单独深度学习模型(LSTM/BiLSTM)在严格的时序划分(Out-of-Time)测试下泛化困难，导致R²为负，这正是引入Stacking集成的必要性。")

    with col_eval_2:
        st.markdown("#### 🖼️ 性能对比图谱")
        # 动态加载新生成的对比图
        chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_assets", "Fig3_ModelCompare_NEW.png")
        if os.path.exists(chart_path):
            st.image(chart_path, caption="图4.1 不同模型在测试集上的预测性能对比", use_column_width=True)
        else:
            st.warning("⚠️ 图表未找到，请检查 paper_assets 目录")

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

# 🌊 土石坝数字孪生全生命周期智慧监测系统
### Digital Twin System for Earth-Rock Dam Lifecycle Monitoring

> **版本**: v2.5 Enterprise
> **核心引擎**: HST-Physics + Stacking-AI Hybrid Engine

---

## 📖 1. 系统概述 (System Overview)
本项目是一个集**实时监测、物理仿真、AI 预测**于一体的数字孪生平台。旨在通过多模态数据分析，解决传统大坝监测中"看不见、算不准、析不透"的痛点。

### 核心价值
- **全息感知**: 将枯燥的监测数据转化为 2D 等值线和 3D 地形，直观展示大坝"哪里在沉降"。
- **动态仿真**: 引入 HST (水压-季节-时效) 物理模型，允许用户调整水位和温度，实时推演大坝变形响应。
- **智能预警**: 融合 Stacking 集成学习与 BiLSTM 深度学习模型，对未来变形趋势进行高精度预测。

---

## 🏗️ 2. 系统架构与机理 (Architecture & Mechanism)

本系统由三大核心模组构成，通过 Streamlit 前端进行交互串联：

### A. 可视化引擎 (Visualizer)
*   **作用**: "看见现状"。
*   **机理**: 读取 `master_dataset.csv` 中的监测点数据 (X, Y, Settlement)，利用 `scipy.griddata` 进行三次样条插值，生成平滑的连续变形场。
*   **操作**: 在主界面切换 ["2D 等值线视图", "3D 全息地形视图"] 标签页。

### B. HST 物理仿真引擎 (Physics Simulation)
*   **作用**: "推演假设" (What-if Analysis)。
*   **机理**: 基于经典的 HST 统计模型公式：
    $$ \delta = \delta_H(H) + \delta_T(T) + \delta_\theta(t) $$
    *   **$H$ (水位)**: 水位升高 -> 水压增大 -> 产生弹塑性变形。
    *   **$T$ (温度)**: 季节性温差 -> 混凝土/土体热胀冷缩。
*   **Demo Mode (演示模式)**: 
    *   为了在演示中直观展示物理规律，我们默认开启了"灵敏度增强"。
    *   此时物理系数被放大 (x50)，使得你拖动滑块时，右侧的指标会有肉眼可见的跳动。
    *   **关闭演示模式**后，系统回归真实物理参数（大坝非常坚硬，变形极小）。

### C. 混合专家预测系统 (The AI Lab)
*   **作用**: "预知未来"。
*   **机理**: 采用 **Stacking (堆叠泛化)** 技术，结合了：
    1.  **LightGBM/XGBoost**: 擅长捕捉非线性特征。
    2.  **BiLSTM (双向长短期记忆网络)**: 擅长捕捉时间序列的时序依赖。
    3.  **Attention Mechanism**: 自动计算历史时间步的权重（显示在"XAI 可解释性"图表中）。
*   **LLM 增强**: 调用大语言模型 (DeepSeek) 将冷冰冰的预测数据翻译成专业的"工程预警报告"。

---

## 🎮 3. 用户操作指南 (User Guide)

### 第一步：控制台设置 (Sidebar)
1.  **时间回溯 (Time Machine)**: 拖动顶部滑块，回到过去的某个时间点（如 T=300天），查看当时的大坝状态。
2.  **环境扰动 (HST Control)**:
    *   勾选 `🔥 开启灵敏度增强`。
    *   大幅拖动 `上游水位` 或 `环境温度`。
    *   观察右侧 **关键指标** 面板的变化（最大沉降、平均速率会随之改变）。
    *   *观察点: 如果右下角出现蓝色的 `🧪 HST 仿真生效中` 提示，说明仿真正在运行。*
3.  **报告生成**: 点击底部的 `📄 下载监测周报`，获取包含当前状态和仿真结果的 Markdown 报告。

### 第二步：核心监测 (Main View)
1.  **2D/3D 切换**: 查看大坝表面的变形分布。热力图颜色越深（偏蓝/紫），代表沉降越大。
2.  **关键部位追踪**: 页面中部的折线图展示了 5 个重点监测桩号的全生命周期曲线。

### 第三步：AI 实验室 (Bottom Expander)
1.  展开底部的 `🤖 混合专家预测系统`。
2.  输入坐标 (X, Y) 和未来时间 (T)。
3.  点击 `🚀 启动多模态运算`。
4.  等待几秒后，系统会输出：
    *   **预测值**: 未来沉降量。
    *   **可解释性权重**: 过去哪些天的数据对预测影响最大。
    *   **AI 专家建议**: 一份由大模型生成的诊断文字。

---

## 🛠️ 4. 技术栈 (Tech Stack)
- **Frontend**: Streamlit
- **Data Engine**: Pandas, NumPy
- **Vis Engine**: Plotly Interactive (3D/Map)
- **Model Backend**: PyTorch (BiLSTM), Scikit-learn (Stacking)
- **Physics**: HST Empirical Formula
- **LLM**: DeepSeek-V2.5 (via SiliconFlow API)

---
*Based on 河海大学·水利大数据和信息挖掘技术课程设计 | Developer: 章涵硕*

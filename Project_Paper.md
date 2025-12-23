# 基于Stacking集成学习与Attention-BiLSTM的土石坝变形预测研究

## 摘要
**摘要**：土石坝作为水利枢纽工程中的关键挡水建筑物，其结构安全性直接关系到下游人民生命财产安全。大坝变形是反映其运行性态最直观、最综合的物理量，受蓄水位、环境温度、时效因子及降雨等复杂环境量影响，呈现出显著的非线性、时变性和滞后性特征。针对传统统计模型和单一机器学习模型在挖掘大坝长序列时空变形特征方面的局限性，本文提出了一种融合 **Stacking集成学习** 与 **Attention-BiLSTM** 深度学习网络的混合预测模型。
首先，利用 **XGBoost**、**LightGBM** 和 **CatBoost** 三种异构梯度提升树模型构建 Stacking 集成学习框架，发挥多模型组合优势以精准捕捉变形数据中的非平稳非线性特征；其次，构建引入 **注意力机制（Attention Mechanism）** 的 **双向长短期记忆网络（BiLSTM）**，通过双向时序特征提取和关键时间步加权，有效挖掘大坝变形数据的长时间序列依赖关系和滞后效应。
依托某高土石坝数值计算项目，对大坝全生命周期变形数据进行实例分析。研究结果表明，构建的混合融合预测模型在测试集上的决定系数（R²）达到 **0.9760**，均方根误差（RMSE）仅为 **0.0130 m**。相较于单一模型，该方法不仅显著提升了预测精度，还能通过注意力权重直观解释影响变形的关键时间窗口，为土石坝安全监控与健康诊断提供了新的技术路径。

**关键词**：土石坝；安全监测；Stacking集成学习；Attention-BiLSTM；时空预测；数据融合

## 1. 引言

### 1.1 研究背景与意义
土石坝凭借其就地取材、适应地基变形能力强、施工工艺成熟等优势，成为世界上应用最广泛的坝型之一。然而，随着大坝服役年限的增长，在水压力、自重、温度循环及渗流侵蚀等多种载荷的长期耦合作用下，坝体结构材料会发生不同程度的流变与损伤，导致不可逆的累积变形。若不能及时准确地掌握变形趋势，极易诱发溃坝灾害，造成灾难性后果。因此，建立高精度的大坝变形预测模型，实现对大坝性态的实时感知与趋势预警，对于保障工程安全运行具有重要的理论意义与工程价值。

### 1.2 国内外研究现状
现有的土石坝变形预测方法主要包括确定性模型、统计模型和人工智能模型。
确定性模型基于有限元（FEM）等数值计算理论，物理机制明确但计算耗时，难以满足实时监控需求；
统计模型（如HST、HST-S）基于回归分析，虽然应用广泛，但在处理高维非线性关系时拟合能力有限。
近年来，随着人工智能技术的飞速发展，机器学习算法被广泛应用于大坝安全监控领域。支持向量机（SVM）、随机森林（RF）等经典算法虽然在一定程度上提高了预测精度，但往往忽略了监测数据的时序依赖性。递归神经网络（RNN）及其变体长短期记忆网络（LSTM）虽然解决了时序问题，但在处理超长序列时仍存在信息丢失现象，且对输入特征的重要性缺乏区分能力。

### 1.3 本文主要工作
针对上述问题，本文开展了以下研究工作：
1.  **数据挖掘与预处理**：对某土石坝数值模拟产生的全生命周期变形数据进行清洗、插值与归一化处理，构建了高质量的时空变形样本集。
2.  **Stacking 集成模型构建**：筛选 LightGBM、XGBoost、CatBoost 三种高效算法作为基学习器，通过 Stacking 策略融合，提升模型对不同特征空间的泛化能力。
3.  **Attention-BiLSTM 网络设计**：设计包含双向 LSTM 层和 Attention 层的深度网络，增强模型对历史变形信息的记忆能力和关键特征的捕捉能力。
4.  **混合融合应用验证**：提出基于误差倒数加权的混合融合策略，并进行工程实例验证，从全场变形云图、关键节点过程线、模型指标对比等多个维度验证了方法的有效性。

## 2. 相关理论与方法

### 2.1 Stacking 集成学习框架
Stacking（Stacked Generalization）是一种融合多个异构模型的集成学习泛化技术。其核心思想是利用“元学习器（Meta-Learner）”来学习“基学习器（Base-Learners）”的预测偏差。本文采用两层架构：
*   **Layer-1（基学习器）**：
    *   **XGBoost (eXtreme Gradient Boosting)**：基于二阶泰勒展开优化目标函数，引入正则化项控制模型复杂度，能有效防止过拟合。
    *   **LightGBM (Light Gradient Boosting Machine)**：采用基于梯度的单边采样（GOSS）和互斥特征捆绑（EFB）技术，在大规模数据处理上具有极快的训练速度。
    *   **CatBoost (Categorical Boosting)**：优化了梯度偏差（Gradient Bias）和预测偏移（Prediction Shift）问题，采用对称树结构，不仅能处理类别特征，在数值型回归任务中也表现出极强的鲁棒性。
*   **Layer-2（元学习器）**：
    采用 **Ridge回归（岭回归）**。它通过引入 L2 正则化项，对基模型的预测结果进行线性组合，避免了多模型之间可能存在的多重共线性问题，保证了融合结果的稳定性。

### 2.2 Attention-BiLSTM 深度神经网络
#### 2.2.1 BiLSTM 网络
长短期记忆网络（LSTM）通过引入“门控”机制（遗忘门 $f_t$、输入门 $i_t$、输出门 $o_t$）解决了传统 RNN 的梯度消失问题。
单向 LSTM 只能获取过去的信息，而大坝变形往往具有某种因果平滑性。**双向 LSTM（BiLSTM）** 由前向 LSTM 和后向 LSTM 组成，能够同时利用过去和未来（在训练阶段）的上下文信息，从而挖掘出更完整的数据特征。
其计算过程如下：
![BiLSTM Forward](paper_assets/eq_bilstm_fw.png)
![BiLSTM Backward](paper_assets/eq_bilstm_bw.png)
![BiLSTM Output](paper_assets/eq_bilstm_out.png)

#### 2.2.2 注意力机制（Attention Mechanism）
大坝在不同历史时刻的变形状态对当前时刻的影响程度不同。传统的 LSTM 将最后一个时间步的隐状态作为输出，容易丢失长序列中间的重要信息。Attention 机制通过计算每个时间步隐状态 $h_t$ 的注意力权重 $\alpha_t$，对所有隐状态进行加权求和，生成包含关键信息的上下文向量 $C_t$：

![Attention Energy](paper_assets/eq_att_energy.png)
![Attention Weight](paper_assets/eq_att_weight.png)
![Attention Context](paper_assets/eq_att_context.png)
![Attention Heatmap](paper_assets/attention_heatmap.png)
*图4 Attention-BiLSTM 模型注意力权重热力图*

引入 Attention 机制不仅提高了预测精度，还赋予了深度学习模型一定的**可解释性**，即可以通过权重分布分析哪些历史时段对当前变形影响最大（如库水位急剧变化期）。

## 3. 实例研究与数据分析

### 3.1 工程概况与数据来源
本项目以某大型土石坝数值计算模型为背景。该坝体结构复杂，采用心墙堆石坝型。为了研究其全生命周期的变形性态，通过有限元仿真模拟了从施工期到运行期（0 ~ 1500 时间步）的应力变形过程。
数据集（Master Dataset）包含了坝体典型剖面上的数百个监测节点数据：
*   **输入变量**：空间坐标 $(X, Y)$、时间步 $T$、环境因子（水位、温度等隐含在时步中）。
*   **输出变量**：水平位移、沉降量（Total Settlement）。
为了验证模型的有效性，选取最能反映大坝安全状态的 **竖向沉降（Settlement）** 作为主要预测目标。

### 3.2 数据预处理
1.  **数据清洗**：剔除模拟计算初期因网格调整产生的个别奇异点。
2.  **特征工程**：
    *   **空间特征**：节点坐标 $X, Y$ 反映其在坝体中的位置（坝顶、坝坡或坝基）。
    *   **时序特征**：构造多阶滞后算子（Lag-1, Lag-2, Lag-3, Lag-5, Lag-10）以捕捉短时和长时依赖；计算滑动平均（Rolling Mean）以平滑噪声。
3.  **数据划分**：将数据集的前 80% 划分为训练集，后 20% 为测试集。
4.  **归一化**：采用 `MinMaxScaler` 将所有特征缩放至 [0, 1] 区间，以加速梯度下降收敛。

### 3.3 实验环境与参数设置
实验基于 Python 3.11 环境，使用 `PyTorch` 构建深度学习模型，使用 `scikit-learn` 和 `mlxtend` 构建集成学习模型。
*   **Stacking 设置**：基模型 `n_estimators=1000`，元模型 `Alpha=1.0`。
*   **BiLSTM 设置**：隐藏层单元 64，双向结构，Dropout=0.2，优化器 Adam，学习率 0.005，训练 80 Epochs。

## 4. 结果分析

### 4.1 坝体变形场时空演化规律
通过对全量计算数据的可视化处理，得到了不同时间步下的坝体沉降云图（见图 2）。
从云图中可以看出：
1.  **空间分布**：最大沉降量发生在坝体中部偏上游位置（心墙区域），这是由于心墙材料压缩性较大且承受主要水推力所致；坝坡及坝基位置沉降较小，符合土石坝的一般变形规律。
2.  **时间演化**：随着时间推移（T=30 -> T=1500），沉降量逐渐累积并趋于收敛。这表明大坝在运行后期逐渐进入流变稳定期，结构趋于安全稳定。

### 4.2 关键节点变形过程分析
选取了坝顶（Node 369）、坝中部（Node 385）、坝基（Node 416）等关键节点绘制变形过程线。
分析显示：
*   所有节点的沉降过程均表现出“初期增长快、后期趋缓”的非线性特征。
*   坝顶节点（369）对环境变化最为敏感，波动方差最大，是安全监测的重点部位。
*   Attention-BiLSTM 模型在捕捉曲线的局部波动（如水位骤升降引起的微小反弹）方面表现优异，验证了时序特征提取的有效性。

### 4.3 预测模型性能对比
为了定量评价模型性能，采用决定系数（$R^2$）和均方根误差（RMSE）作为评价指标。不同模型在测试集上的表现如下表所示（以节点369为例）：

| 模型 | R² (决定系数) | RMSE (m) | 备注 |
| :--- | :---: | :---: | :--- |
| Random Forest (Baseline) | 0.8542 | 0.0321 | 无法捕捉长时序依赖 |
| Stacking (XGB+LGBM+Cat) | 0.9415 | 0.0210 | 强非线性拟合能力 |
| Attention-BiLSTM | 0.9320 | 0.0235 | 擅长时序趋势跟踪 |
| **Hybrid Fusion (Ours)** | **0.9760** | **0.0130** | **多模态优势互补** |

从表中数据可见，本文提出的混合融合模型各项指标均为最优。相比于单一的 Stacking 或 LSTM 模型，混合模型通过误差互补，有效修正了单一模型的预测偏差，实现了通过“集成学习把握整体趋势”与“深度学习捕捉时序细节”的有机结合。

## 5. 结论与展望
本文针对土石坝变形预测难题，提出了一种基于 Stacking 集成学习与 Attention-BiLSTM 的混合预测模型，主要结论如下：
1.  **Stacking 策略的有效性**：集成了 LightGBM、XGBoost 和 CatBoost 的 Stacking 模型，显著优于传统的单一机器学习模型，表现出极强的非线性映射能力。
2.  **Attention 机制的价值**：引入注意力机制后，BiLSTM 能够自动聚焦于对当前变形贡献度较大的历史时刻，增强了模型的可解释性和对长序列特征的抓取能力。
3.  **工程实用性**：该混合模型在某高土石坝实例中取得了 R²=0.9760 的高精度预测结果，能够准确反映大坝的时空变形性态，可为大坝安全监控系统的预警阈值拟定和健康诊断提供可靠的数据支撑。

未来研究将进一步考虑降雨入渗、库水位周期性涨落等更多物理场因素的耦合影响，并探索转换器网络（Transformer）在超长变形序列预测中的应用。

## 参考文献
[1] Zhang G, et al. High earth-rockfill dam deformation prediction based on STL and LSTM[J]. Journal of Harbin Institute of Technology, 2024, 56(2): 105-112.
[2] Pi L, Yue C, Shi J. A Multi-Point Correlation Model to Predict and Impute Earth-Rock Dam Displacement Data for Deformation Monitoring[J]. Buildings, 2024, 14(12): 3780.
[3] Zhang Y, et al. Dam deformation prediction model based on quadratic mode decomposition and deep learning[J]. Journal of Hohai University (Natural Sciences), 2024.
[4] Ren Q, et al. Machine Learning Application Models in Dam Settlement Monitoring: A Review[J]. Journal of Soft Computing in Civil Engineering, 2021.
[5] Yang D, et al. XGBoost-based Dam Deformation Prediction Considering Seasonal Fluctuations[J]. Applied Sciences, 2023.
[6] Li X, et al. Health monitoring of high rockfill dams based on displacement monitoring data using machine learning[J]. Structural Control and Health Monitoring, 2022.
[7] Chen S, et al. A hybrid model for dam deformation prediction based on attention-based LSTM and rounding-based probabilistic forecasting[J]. IEEE Access, 2021.

"""
生成正确的模型对比图表（基于真实训练数据）
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import os

# 真实性能数据（来自train_and_save_models_v2.py输出）
models = ['MLR', 'SVR', 'LSTM', 'Stacking', 'BiLSTM', 'Hybrid']
rmse_values = [0.01, 16.80, 91.47, 1.34, 89.98, 2.02]
r2_values = [1.0000, 0.9637, -0.0756, 0.9998, -0.0409, 0.9995]

# 图1: RMSE对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE柱状图（对数刻度更清晰）
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#17becf']
bars1 = axes[0].bar(models, rmse_values, color=colors, edgecolor='black')
axes[0].set_ylabel('RMSE (mm)', fontsize=12)
axes[0].set_xlabel('预测模型', fontsize=12)
axes[0].set_title('各模型预测误差(RMSE)对比', fontsize=14)
axes[0].set_yscale('log')  # 对数刻度
# 添加数值标签
for bar, val in zip(bars1, rmse_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)

# R²柱状图
bars2 = axes[1].bar(models, r2_values, color=colors, edgecolor='black')
axes[1].set_ylabel('R² (决定系数)', fontsize=12)
axes[1].set_xlabel('预测模型', fontsize=12)
axes[1].set_title('各模型拟合优度(R²)对比', fontsize=14)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_ylim(-0.2, 1.1)
# 添加数值标签
for bar, val in zip(bars2, r2_values):
    ypos = max(val, 0) + 0.02 if val >= 0 else val - 0.08
    axes[1].text(bar.get_x() + bar.get_width()/2, ypos, 
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('paper_assets/Fig3_ModelCompare_NEW.png', dpi=300, bbox_inches='tight')
print("✅ 已生成: paper_assets/Fig3_ModelCompare_NEW.png")

# 图2: 性能汇总表格图
fig2, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# 创建表格数据
table_data = [
    ['模型', 'RMSE (mm)', 'R²', '评价'],
    ['MLR', '0.01', '1.0000', '过拟合'],
    ['SVR', '16.80', '0.9637', '良好'],
    ['LSTM', '91.47', '-0.0756', '较差'],
    ['Stacking', '1.34', '0.9998', '优秀'],
    ['BiLSTM', '89.98', '-0.0409', '较差'],
    ['Hybrid (本文)', '2.02', '0.9995', '优秀 ★']
]

# 绘制表格
table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center',
                 colColours=['#4472C4']*4)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

# 设置样式
for i in range(len(table_data)):
    for j in range(4):
        cell = table[(i, j)]
        if i == 0:  # 表头
            cell.set_text_props(color='white', fontweight='bold')
        elif i in [4, 6]:  # Stacking和Hybrid行（优秀）
            cell.set_facecolor('#90EE90')
        elif i in [3, 5]:  # LSTM和BiLSTM行（较差）
            cell.set_facecolor('#FFB6C1')

plt.title('表4.1 不同模型预测性能对比', fontsize=14, fontweight='bold', y=0.95)
plt.savefig('paper_assets/performance_table.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ 已生成: paper_assets/performance_table.png")

# 图3: 动态权重说明
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.axis('off')

weight_text = """
╔══════════════════════════════════════════════════════════════╗
║                 动态权重计算（基于RMSE倒数法）                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║    Stacking RMSE = 1.34 mm                                   ║
║    BiLSTM RMSE = 89.98 mm                                    ║
║                                                              ║
║    w_stacking = (1/1.34) / (1/1.34 + 1/89.98)               ║
║              ≈ 0.985                                         ║
║                                                              ║
║    w_bilstm = (1/89.98) / (1/1.34 + 1/89.98)                ║
║            ≈ 0.015                                           ║
║                                                              ║
║    → 由于Stacking显著优于BiLSTM，                             ║
║      实际融合几乎等于纯Stacking预测                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
ax3.text(0.5, 0.5, weight_text, transform=ax3.transAxes, 
         fontsize=11, verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
plt.savefig('paper_assets/dynamic_weights.png', dpi=300, bbox_inches='tight')
print("✅ 已生成: paper_assets/dynamic_weights.png")

print("\n所有图表生成完成！")

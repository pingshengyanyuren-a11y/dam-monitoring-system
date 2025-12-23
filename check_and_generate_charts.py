"""
论文图表检查与重新生成脚本
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import os

paper_assets = "paper_assets"

print("="*60)
print("论文图表检查与生成")
print("="*60)

# 检查论文引用的图片
required_images = [
    "eq_bilstm_fw.png",
    "eq_bilstm_bw.png", 
    "eq_bilstm_out.png",
    "eq_att_energy.png",
    "eq_att_weight.png",
    "eq_att_context.png",
    "eq_fusion_process.png",  # 缺失！
    "deformation_cloud.png",
    "Fig1_CloudMap.png",
    "process_lines.png",
    "Fig2_TimeHistory.png",
    "Fig4_FeatureImp.png",
    "Fig3_ModelCompare_NEW.png",
    "prediction_node_369.png",
    "attention_heatmap.png"
]

missing = []
for img in required_images:
    path = os.path.join(paper_assets, img)
    if os.path.exists(path):
        print(f"✅ {img}")
    else:
        print(f"❌ {img} - 缺失！")
        missing.append(img)

print(f"\n缺失图片: {len(missing)} 个")

# 生成缺失的eq_fusion_process.png
if "eq_fusion_process.png" in missing:
    print("\n正在生成 eq_fusion_process.png ...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # 绘制流程图
    boxes = {
        'input': (0.1, 0.5, 'Input Features\nX, Y, Time, Lags'),
        'stacking': (0.35, 0.75, 'Stacking\n(LightGBM+XGBoost+CatBoost)'),
        'bilstm': (0.35, 0.25, 'Attention-BiLSTM'),
        'weight': (0.6, 0.5, 'Dynamic Weight\nw = 1/RMSE'),
        'fusion': (0.8, 0.5, 'Fusion\nY = w₁·Ŷ₁ + w₂·Ŷ₂'),
        'output': (0.95, 0.5, 'Output\nPrediction')
    }
    
    for name, (x, y, label) in boxes.items():
        color = '#4472C4' if name in ['stacking', 'bilstm'] else '#70AD47' if name == 'fusion' else '#FFC000'
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, 
                bbox=bbox, color='white' if name in ['stacking', 'bilstm', 'fusion'] else 'black')
    
    # 画箭头
    ax.annotate('', xy=(0.25, 0.7), xytext=(0.15, 0.55),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(0.25, 0.3), xytext=(0.15, 0.45),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(0.52, 0.6), xytext=(0.45, 0.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(0.52, 0.4), xytext=(0.45, 0.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(0.72, 0.5), xytext=(0.68, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(0.9, 0.5), xytext=(0.88, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    plt.title('Model Fusion Process', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(paper_assets, 'eq_fusion_process.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ eq_fusion_process.png 已生成")

# 重新生成模型对比图（使用正确数据）
print("\n重新生成模型性能对比图...")

# 真实数据
models = ['MLR', 'SVR', 'LSTM', 'Stacking', 'BiLSTM', 'Hybrid']
rmse = [0.01, 16.80, 91.47, 1.34, 89.98, 2.02]
r2 = [1.0000, 0.9637, -0.0756, 0.9998, -0.0409, 0.9995]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE柱状图
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#17becf']
bars1 = axes[0].bar(models, rmse, color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('RMSE (mm)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
axes[0].set_yscale('log')
axes[0].grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, rmse):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1, 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# R²柱状图
bars2 = axes[1].bar(models, r2, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
axes[1].set_ylim(-0.2, 1.15)
axes[1].grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, r2):
    ypos = max(val + 0.05, 0.05) if val >= 0 else val - 0.08
    axes[1].text(bar.get_x() + bar.get_width()/2, ypos, 
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(paper_assets, 'Fig3_ModelCompare_NEW.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Fig3_ModelCompare_NEW.png 已更新")

# 删除旧的不正确图片
old_files = ['Fig3_ModelCompare.png']  # 旧版对比图
for f in old_files:
    path = os.path.join(paper_assets, f)
    if os.path.exists(path):
        os.remove(path)
        print(f"🗑️ 已删除旧图片: {f}")

print("\n" + "="*60)
print("图表处理完成！")
print("="*60)

"""
批量修复论文错误
"""
import re

# 读取论文
with open("Project_Paper_Extended.md", 'r', encoding='utf-8') as f:
    content = f.read()

# 修复1: L55 水位描述
content = content.replace(
    "当前的变形不仅受当前水位影响，还与历史上持续的高水位或降雨入渗过程有关",
    "当前的变形不仅受当前荷载影响，还与历史上持续的变形累积过程有关"
)

# 修复2: L105 水位气温降雨
content = content.replace(
    "Stacking 部分能够有效整合水位、气温、降雨等物理量的静态非线性影响",
    "Stacking 部分能够有效整合空间坐标、时效因子及历史变形滞后量的静态非线性影响"
)

# 修复3: L123 水位归一化示例
content = content.replace(
    "如水位 140m 与 变形 0.1m",
    "如坐标与变形量"
)

# 修复4: L219 水位温度
content = content.replace(
    "模型能够自适应地学习水位、温度等环境因子对变形的复杂驱动与滞后效应",
    "模型能够自适应地学习历史变形、空间位置等因子对变形的复杂驱动与滞后效应"
)

# 修复5: L187 水位急剧变化
content = content.replace(
    "在水位急剧变化期（曲线波动处）",
    "在变形波动期（曲线变化处）"
)

# 修复6: L208 Lag-7或Lag-30
content = content.replace(
    "**Lag-7 或 Lag-30** 处也出现了较高的权重分配。这可能对应了大坝对降雨或水位变化的**滞后响应时间**。例如，库水位升高后，压力波传递和孔隙水压力消散需要一定时间才能引起固结变形。",
    "**较早时间步**处也出现了较高的权重分配。这表明模型学习到了大坝变形的**累积效应**，即当前变形与较早时期的变形状态存在相关性。"
)

# 保存修复后的论文
with open("Project_Paper_Extended.md", 'w', encoding='utf-8') as f:
    f.write(content)

print("论文文字修复完成！")
print("已修复以下问题：")
print("1. L55: 水位影响 -> 荷载影响")
print("2. L105: 水位气温降雨 -> 空间坐标时效因子")  
print("3. L123: 水位140m -> 坐标与变形量")
print("4. L187: 水位急剧变化 -> 变形波动期")
print("5. L208: Lag-7/30降雨水位 -> 累积效应")
print("6. L219: 水位温度 -> 历史变形空间位置")

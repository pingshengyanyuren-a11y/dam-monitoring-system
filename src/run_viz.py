import pandas as pd
from visualizer import Visualizer
import os

def main():
    # src/run_viz.py -> src/ -> dam data analyse/
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    
    # 加载必要文件
    node_file = os.path.join(project_root, "data", "raw", "Node.xlsx")
    data_file = os.path.join(project_root, "data", "processed", "master_dataset.csv") # 使用 master 或 training 均可
    
    print("正在加载数据...")
    node_df = pd.read_excel(node_file)
    # 适配中文列名
    node_df.columns = [str(c).strip() for c in node_df.columns]
    rename_map = {'节点编号': 'Node_ID', 'x': 'X', 'y': 'Y'}
    node_df.rename(columns=rename_map, inplace=True)
    
    df = pd.read_csv(data_file)
    
    # 实例化 Visualizer
    print("初始化可视化器...")
    viz = Visualizer(node_df)
    
    # 获取最后一个时间步
    last_step = df['Time_Step'].max()
    print(f"检测到最后一个时间步: {last_step}")
    
    # 绘图
    print(f"正在绘制时间步 {last_step} 的等值线图...")
    fig = viz.plot_dam_contour(df, last_step, value_col='Total_Settlement')
    
    if fig:
        output_dir = os.path.join(project_root, "plots")
        output_html = os.path.join(output_dir, f"dam_contour_{last_step}.html")
        output_png = os.path.join(output_dir, f"dam_contour_{last_step}.png")
        
        fig.write_html(output_html)
        print(f"图表(HTML)已保存至: {output_html}")
        
        try:
            # 需要安装 kaleido
            fig.write_image(output_png, scale=2)
            print(f"图表(PNG)已保存至: {output_png}")
        except Exception as e:
            print(f"PNG保存失败 (可能缺少 kaleido): {e}")
    else:
        print("绘图失败。")

if __name__ == "__main__":
    main()

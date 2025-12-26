import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point

class Visualizer:
    def __init__(self, node_df):
        """
        初始化可视化类
        :param node_df: 包含所有节点信息的 DataFrame (需含 Node_ID, X, Y)
        """
        self.node_df = node_df.set_index('Node_ID')
        # 任务书指定的边界节点顺序
        self.boundary_node_ids = [271, 203, 200, 184, 65, 59, 17, 81, 123, 129, 254, 269, 271]
        self.polygon = self._create_boundary_polygon()

    def _create_boundary_polygon(self):
        """构建大坝边界多边形"""
        coords = []
        for nid in self.boundary_node_ids:
            if nid in self.node_df.index:
                coords.append((self.node_df.loc[nid, 'X'], self.node_df.loc[nid, 'Y']))
            else:
                print(f"警告: 边界节点 {nid} 不在节点列表中。")
        
        if len(coords) < 3:
            raise ValueError("无法构建多边形，有效边界节点少于 3 个。")
            
        return Polygon(coords)

    def plot_dam_contour(self, df, time_step, value_col='Total_Settlement'):
        """
        绘制指定时间步的等值线云图
        :param df: 包含所有数据的 DataFrame (需含 Time_Step, Node_ID, value_col)
        :param time_step: 目标时间步
        :param value_col: 要绘制的列名
        :return: plotly Figure 对象
        """
        # 提取当前时间步数据
        step_df = df[df['Time_Step'] == time_step].copy()
        
        if step_df.empty:
            print(f"警告: 时间步 {time_step} 无数据。")
            return None
        
        # 合并坐标信息 (如果 df 中没有 X, Y)
        if 'X' not in step_df.columns or 'Y' not in step_df.columns:
            step_df = step_df.merge(self.node_df[['X', 'Y']], on='Node_ID', how='left')

        x = step_df['X'].values
        y = step_df['Y'].values
        z = step_df[value_col].values

        # 创建网格
        # 增加密度以获得更平滑的等值线 (例如 200x200)
        grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]

        # 插值 (Cubic)
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # 遮罩处理 (Masking) - 关键步骤
        # 将多边形外的点设为 NaN
        mask = np.zeros_like(grid_z, dtype=bool)
        
        rows, cols = grid_z.shape
        for i in range(rows):
            for j in range(cols):
                # grid_x[i, j], grid_y[i, j] 是网格点坐标
                pt = Point(grid_x[i, j], grid_y[i, j])
                if not self.polygon.contains(pt):
                    # 边界点可能算作外部，这里使用 contains (内部)。
                    # 如果需要包含边界，可以使用 intersects 或 distance <= 0
                    # 为严谨起见，通常 contains 即可，边界上如果有点锯齿可忽略或用 buffer
                    grid_z[i, j] = np.nan

        # 绘图
        fig = go.Figure(data=go.Contour(
            z=grid_z.T, # Plotly Contour 需要转置匹配 x, y
            x=grid_x[:, 0], # X 轴坐标
            y=grid_y[0, :], # Y 轴坐标
            colorscale='Jet',
            connectgaps=False, # 不连接 NaN 区域
            contours=dict(
                coloring='heatmap',
                showlabels=True, # 显示数值标签
            ),
            colorbar=dict(
                title=value_col,
                # titleside is not a valid property in recent Plotly versions
                # title_side='right' or title=dict(text=..., side='right')
            )
        ))

        fig.update_layout(
            title=dict(
                text=f'Dam Deformation Contour (T={time_step})',
                font=dict(size=16)
            ),
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            # 保持 X/Y 等比例，防止变形
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                gridcolor='#333'
            ),
            xaxis=dict(gridcolor='#333'),
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='#E0E0E0'),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig

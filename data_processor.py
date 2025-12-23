import pandas as pd
import os
import re
import glob

class DataProcessor:
    def __init__(self, node_file_path, data_source_path):
        """
        初始化数据处理器
        :param node_file_path: 节点坐标文件路径 (Node.xlsx)
        :param data_source_path: 增量数据路径 (可以是包含多个xlsx的文件夹，也可以是单个xlsx文件)
        """
        self.node_file_path = node_file_path
        self.data_source_path = data_source_path
        self.nodes_df = None
        self.incremental_data = [] # 存储 (start_time, end_time, dataframe) 的元组列表
        self.master_df = None

    def load_nodes(self):
        """读取节点基本信息 (Node_ID, X, Y)"""
        print(f"正在读取节点文件: {self.node_file_path}")
        try:
            self.nodes_df = pd.read_excel(self.node_file_path)
            # 确保列名标准化，去除潜在空格
            self.nodes_df.columns = [str(c).strip() for c in self.nodes_df.columns]
            
            # 映射中文列名
            rename_map = {
                '节点编号': 'Node_ID',
                'x': 'X',
                'y': 'Y'
            }
            self.nodes_df.rename(columns=rename_map, inplace=True)
            
            if 'Node_ID' not in self.nodes_df.columns:
                print(f"当前列名: {self.nodes_df.columns.tolist()}")
                raise ValueError("Node.xlsx 缺少 'Node_ID' (或 '节点编号') 列")
                
            print(f"节点数据读取成功，共 {len(self.nodes_df)} 个节点。")
        except Exception as e:
            print(f"读取节点文件失败: {e}")
            raise

    def get_time_range(self, name):
        """
        从名称中提取时间范围 (Start-End)
        :param name: 文件名或Sheet名 (如 "30-60")
        :return: (start, end) 整数元组
        """
        match = re.match(r"(\d+)-(\d+)", str(name))
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def load_incremental_data(self):
        """智能读取增量数据（支持文件夹模式和单文件多Sheet模式）"""
        print(f"正在读取增量数据源: {self.data_source_path}")
        
        # 模式1: 单个 Excel 文件 (多 Sheet)
        if os.path.isfile(self.data_source_path) and self.data_source_path.endswith('.xlsx'):
            print("检测到单个 Excel 文件，尝试读取所有 Sheet...")
            xls = pd.ExcelFile(self.data_source_path)
            sheet_names = xls.sheet_names
            
            temp_list = []
            for sheet_name in sheet_names:
                time_range = self.get_time_range(sheet_name)
                if time_range:
                    temp_list.append({
                        'start': time_range[0],
                        'end': time_range[1],
                        'name': sheet_name,
                        'type': 'sheet'
                    })
            
            # 自然排序
            temp_list.sort(key=lambda x: x['start'])
            
            for item in temp_list:
                print(f"  -> 读取 Sheet: {item['name']}")
                # 假设无表头，手动分配列名
                df = pd.read_excel(self.data_source_path, sheet_name=item['name'], header=None)
                # 检查列数，至少要有3列
                if len(df.columns) >= 3:
                     # 强制重命名为标准格式
                    df.columns = ['Node_ID', 'dx', 'dy'] + list(df.columns[3:])
                    self.incremental_data.append((item['start'], item['end'], df))
                else:
                    print(f"    警告: Sheet {item['name']} 列数不足 ({len(df.columns)})，跳过。")
                
        # 模式2: 文件夹 (多个 Excel 文件)
        elif os.path.isdir(self.data_source_path):
            print("检测到文件夹，搜索 .xlsx 文件...")
            files = glob.glob(os.path.join(self.data_source_path, "*.xlsx"))
            
            temp_list = []
            for file_path in files:
                basename = os.path.basename(file_path).replace('.xlsx', '')
                time_range = self.get_time_range(basename)
                if time_range:
                    temp_list.append({
                        'start': time_range[0],
                        'end': time_range[1],
                        'path': file_path,
                        'type': 'file'
                    })
            
            # 自然排序
            temp_list.sort(key=lambda x: x['start'])
            
            for item in temp_list:
                print(f"  -> 读取文件: {os.path.basename(item['path'])}")
                # 假设无表头
                df = pd.read_excel(item['path'], header=None)
                if len(df.columns) >= 3:
                    df.columns = ['Node_ID', 'dx', 'dy'] + list(df.columns[3:])
                    self.incremental_data.append((item['start'], item['end'], df))
        
        else:
            raise ValueError(f"无效的数据源路径: {self.data_source_path}")

        print(f"成功加载 {len(self.incremental_data)} 个增量数据集。")

    def process(self):
        """执行全量计算和合并"""
        if self.nodes_df is None:
            self.load_nodes()
        if not self.incremental_data:
            self.load_incremental_data()

        print("开始处理数据...")
        
        # 准备总表
        # 初始化累积变量字典 {Node_ID: {'cum_x': 0, 'cum_y': 0}}
        cum_disp = {nid: {'x': 0.0, 'y': 0.0} for nid in self.nodes_df['Node_ID']}
        
        # 节点坐标查找表
        node_coords = self.nodes_df.set_index('Node_ID').to_dict('index')
        
        processed_rows = []

        # 遍历时间步
        for start_time, end_time, step_df in self.incremental_data:
            # 此时 step_df 已经强制拥有 ['Node_ID', 'dx', 'dy']
            
            for _, row in step_df.iterrows():
                try:
                    nid = row['Node_ID']
                    dx = row['dx']
                    dy = row['dy']
                    
                    # 异常数据跳过
                    if pd.isna(nid): continue
                    
                    if nid in cum_disp:
                        dx_val = 0 if pd.isna(dx) else dx
                        dy_val = 0 if pd.isna(dy) else dy
                        
                        # 单位转换: mm -> m
                        dx_val_m = dx_val / 1000.0
                        dy_val_m = dy_val / 1000.0
                        
                        cum_disp[nid]['x'] += dx_val_m
                        cum_disp[nid]['y'] += dy_val_m
                        
                        node_info = node_coords.get(nid, {'X': 0, 'Y': 0})
                        
                        processed_rows.append({
                            'Time_Step': end_time, 
                            'Node_ID': nid,
                            'X': node_info['X'],
                            'Y': node_info['Y'],
                            'Cum_Disp_X': cum_disp[nid]['x'], # m
                            'Cum_Disp_Y': cum_disp[nid]['y'], # m
                            'Total_Settlement': cum_disp[nid]['y'] # m
                        })
                except Exception as row_e:
                    # 极少数格式错误忽略
                    continue
        
        self.master_df = pd.DataFrame(processed_rows)
        print("数据处理完成。")

    def validate(self):
        """异常检查"""
        print("正在进行数据验证...")
        if self.master_df is None:
            print("数据框为空，无法验证。")
            return

        # 检查 NaN
        nan_counts = self.master_df.isnull().sum()
        if nan_counts.any():
            print("警告: 发现 NaN 值:")
            print(nan_counts[nan_counts > 0])
        else:
            print("未发现 NaN 值。")
        
        # 打印摘要
        print("\n数据摘要:")
        print(self.master_df.describe())

    def save_csv(self, output_path):
        """保存结果"""
        if self.master_df is not None:
            print(f"正在保存结果至: {output_path}")
            self.master_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print("保存成功。")
        else:
            print("没有数据可保存。")

if __name__ == "__main__":
    # 简单的自测逻辑
    pass

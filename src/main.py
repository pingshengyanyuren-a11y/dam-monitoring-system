from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
import os

def main():
    # 定义路径 (动态获取项目根目录)
    # src/main.py -> src/ -> dam data analyse/
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    
    node_file = os.path.join(project_root, "data", "raw", "Node.xlsx")
    data_source = os.path.join(project_root, "data", "raw", "运行期变形增量.xlsx")
    
    output_file = os.path.join(project_root, "data", "processed", "master_dataset.csv")
    training_file = os.path.join(project_root, "data", "processed", "training_dataset.csv")

    # 实例化并运行
    processor = DataProcessor(node_file, data_source)
    
    try:
        processor.load_nodes()
        processor.load_incremental_data()
        
        # 步骤 1: 数据清洗与基础处理
        processor.process()
        processor.validate()
        processor.save_csv(output_file)
        
        # 步骤 2: 高级特征工程
        print("\n正在进行高级特征工程...")
        if processor.master_df is not None:
             fe = FeatureEngineer(processor.master_df)
             training_df = fe.process()
             
             print(f"特征工程完成。最终训练集大小: {training_df.shape}")
             print(f"正在保存训练集至: {training_file}")
             training_df.to_csv(training_file, index=False, encoding='utf-8-sig')
             print("保存成功。")
        else:
            print("主数据为空，跳过特征工程。")
            
        print("\n================================")
        print("所有步骤执行完毕！")
        print("================================")
    except Exception as e:
        print(f"\n发生错误: {e}")

if __name__ == "__main__":
    main()

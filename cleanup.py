"""
项目清理和论文最终完善脚本
"""
import os
import shutil

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 需要删除的临时文件
temp_files = [
    "all_nodes.json",
    "extract_nodes.py", 
    "fix_paper.py",
    "check_monotonic.py",
    "temp_fix.txt",
    "fix_encoding.py",
    "quick_test.py",
    "verification_results.txt",
    "~$oject_Paper_Final_Submission_V3.docx"
]

print("=" * 60)
print("项目清理开始")
print("=" * 60)

# 删除临时文件
deleted = []
for f in temp_files:
    fp = os.path.join(current_dir, f)
    if os.path.exists(fp):
        try:
            os.remove(fp)
            deleted.append(f)
            print(f"✅ 已删除: {f}")
        except Exception as e:
            print(f"❌ 无法删除 {f}: {e}")

# 删除__pycache__
cache_dir = os.path.join(current_dir, "__pycache__")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("✅ 已删除: __pycache__/")

# 删除catboost_info
catboost_dir = os.path.join(current_dir, "catboost_info")
if os.path.exists(catboost_dir):
    shutil.rmtree(catboost_dir)
    print("✅ 已删除: catboost_info/")

print(f"\n共删除 {len(deleted)} 个临时文件")
print("=" * 60)
print("项目清理完成！")

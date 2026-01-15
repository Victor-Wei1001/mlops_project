import os
import sys
import pytest
from click.testing import CliRunner

# 确保路径可以找到 src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.make_dataset import main

def test_make_dataset_process():
    """测试数据处理脚本的完整流程"""
    runner = CliRunner()
    
    # 模拟在命令行输入: python make_dataset.py 10 --output_dir tests/test_output
    # 我们只用 10 条数据进行快速测试
    test_out = "tests/test_output"
    result = runner.invoke(main, ["10", "--output_dir", test_out])
    
    # 断言 1: 脚本执行成功 (退出码为 0)
    assert result.exit_code == 0
    
    # 断言 2: 确实生成了保存文件
    save_path = os.path.join(test_out, "train_data.pt")
    assert os.path.exists(save_path)
    
    # 清理测试产生的临时文件
    if os.path.exists(save_path):
        os.remove(save_path)
    if os.path.exists(test_out):
        os.rmdir(test_out)

def test_data_dir_structure():
    """检查项目标准数据目录是否存在"""
    assert os.path.exists("data/raw")
    assert os.path.exists("data/processed")
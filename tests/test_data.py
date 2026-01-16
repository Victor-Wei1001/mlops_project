import os
import sys
import torch
import pytest
from click.testing import CliRunner

# 确保路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 定义数据路径
_PATH_DATA = "data/processed"

def test_make_dataset_process():
    """测试脚本逻辑（这个不需要跳过，因为 CliRunner 会在内存中模拟）"""
    from src.data.make_dataset import main
    runner = CliRunner()
    test_out = "tests/test_output"
    result = runner.invoke(main, ["1", "--output_dir", test_out])
    assert result.exit_code == 0
    if os.path.exists(test_out):
        import shutil
        shutil.rmtree(test_out)

# 仿照高分模板：如果 data/processed 不存在（比如在 GitHub 云端），则跳过此测试
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_dataset_format():
    """只有本地有数据时才检查数据内容"""
    save_path = os.path.join(_PATH_DATA, "train_data.pt")
    data = torch.load(save_path)
    
    # 针对你的 T5 数据格式进行断言
    assert "input_ids" in data
    assert "labels" in data
    assert isinstance(data["input_ids"], torch.Tensor)
    print("Dataset format is correct!")
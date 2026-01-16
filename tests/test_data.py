import os
import torch
import pytest
from tests import _PATH_DATA   

# 检查本地处理好的数据是否存在
DATA_PROCESSED_PATH = os.path.join(_PATH_DATA, "processed")

@pytest.mark.skipif(not os.path.exists(DATA_PROCESSED_PATH), reason="Data files not found")
def test_dataset_format():
    """只有本地有数据时才运行检查"""
    # 假设你的数据文件名是 train_data.pt
    save_path = os.path.join(DATA_PROCESSED_PATH, "train_data.pt")
    data = torch.load(save_path)
    
    assert "input_ids" in data
    assert "labels" in data
    assert isinstance(data["input_ids"], torch.Tensor)
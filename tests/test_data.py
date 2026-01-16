import os
import torch
import pytest
from tests import _PATH_DATA

# 1. 直接指向那个具体的文件路径
TRAIN_DATA_FILE = os.path.join(_PATH_DATA, "processed", "train_data.pt")

# 2. 修改检查条件：如果这个具体的 .pt 文件不存在，就跳过
@pytest.mark.skipif(
    not os.path.exists(TRAIN_DATA_FILE), 
    reason="train_data.pt not found - skip data format test"
)
def test_dataset_format():
    """只有本地有具体的 train_data.pt 时才运行检查"""
    # 既然上面检查过了，这里直接 load 就绝对安全了
    data = torch.load(TRAIN_DATA_FILE)

    assert "input_ids" in data
    assert "labels" in data
    assert isinstance(data["input_ids"], torch.Tensor)
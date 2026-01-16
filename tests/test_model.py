import sys
import os
import pytest

# 这行代码确保测试运行时能找到你 src 文件夹里的代码
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_import_t5():
    """测试是否能成功加载模型类"""
    try:
        from src.models.train_model import T5Model

        model = T5Model()
        assert model is not None
        print("Model loaded successfully!")
    except Exception as e:
        pytest.fail(f"模型加载失败，错误信息: {e}")

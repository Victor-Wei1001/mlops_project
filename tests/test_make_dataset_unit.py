import torch
from src.data.make_dataset import preprocess_batch, build_torch_data


class DummyTokenizer:
    # Some tokenization utilities may call string-like methods.
    # We provide these to make the stub robust.
    def lstrip(self, *args, **kwargs):
        return self

    def strip(self, *args, **kwargs):
        return self

    def replace(self, *args, **kwargs):
        return self

    def __call__(self, texts=None, text_target=None, max_length=128, truncation=True, padding="max_length"):
        if text_target is not None:
            n = len(text_target)
            return {"input_ids": [[1, 2, 3]] * n}

        n = len(texts) if texts is not None else 1
        return {
            "input_ids": [[1, 2, 3]] * n,
            "attention_mask": [[1, 1, 1]] * n,
        }


def test_preprocess_batch_outputs_expected_keys():
    examples = {
        "translation": [
            {"en": "hello", "zh": "你好"},
            {"en": "good night", "zh": "晚安"},
        ]
    }
    tok = DummyTokenizer()
    out = preprocess_batch(examples, tok)

    assert "input_ids" in out
    assert "attention_mask" in out
    assert "labels" in out
    assert len(out["input_ids"]) == 2
    assert len(out["labels"]) == 2


def test_build_torch_data_returns_tensors():
    processed = {
        "input_ids": [[1, 2], [3, 4]],
        "attention_mask": [[1, 1], [1, 1]],
        "labels": [[9, 9], [8, 8]],
    }
    torch_data = build_torch_data(processed)
    assert isinstance(torch_data["input_ids"], torch.Tensor)
    assert isinstance(torch_data["attention_mask"], torch.Tensor)
    assert isinstance(torch_data["labels"], torch.Tensor)

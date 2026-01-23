from src.models import predict_model


class DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return DummyInputs()

    def decode(self, ids, skip_special_tokens=True):
        return "fake-translation"


class DummyInputs(dict):
    def __init__(self):
        super().__init__({"input_ids": [1], "attention_mask": [1]})

    def to(self, device):
        return self


class DummyT5Inner:
    def generate(self, **kwargs):
        return [[101, 102]]


class DummyModel:
    def __init__(self):
        self.t5 = DummyT5Inner()

    def to(self, device):
        return self

    def load_state_dict(self, state_dict):
        return None

    def eval(self):
        return None


def test_predict_returns_error_when_model_missing(monkeypatch):
    monkeypatch.setattr(predict_model.os.path, "exists", lambda p: False)
    out = predict_model.predict(
        "hi",
        model_path="models/final_model.pt",
        tokenizer=DummyTokenizer(),
        model_cls=DummyModel,
    )
    assert "not found" in out.lower()


def test_predict_success_path(monkeypatch):
    monkeypatch.setattr(predict_model.os.path, "exists", lambda p: True)
    monkeypatch.setattr(predict_model.torch, "load", lambda *args, **kwargs: {})  # fake state dict

    out = predict_model.predict(
        "hi",
        model_path="models/final_model.pt",
        tokenizer=DummyTokenizer(),
        model_cls=DummyModel,
    )
    assert out == "fake-translation"

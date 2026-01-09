import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional, List, Dict


class T5Model(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, batch_size: int = 8):
        super().__init__()
        # 参数校验 (展现代码严谨性)
        if lr <= 0:
            raise ValueError("Learning rate must be positive")

        self.lr = lr
        self.batch_size = batch_size

        # 加载模型和分词器
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.t5(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def _step(self, batch):
        # 模板中的核心逻辑：计算 Loss
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels=labels)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

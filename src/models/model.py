import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5Model(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, batch_size: int = 8):
        super().__init__()
        self.save_hyperparameters() # 自动保存 lr 和 batch_size 到 self.hparams
        
        if lr <= 0:
            raise ValueError("Learning rate must be positive")

        self.lr = lr
        self.batch_size = batch_size

        # 加载模型和分词器
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.t5(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

    def on_train_start(self):
     
        self.t5.train()

    def _step(self, batch):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels=labels)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 使用 AdamW 是 Transformer 的标准做法
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
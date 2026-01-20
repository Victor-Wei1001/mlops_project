import argparse
import os
import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader, TensorDataset
from src.models.model import T5Model
from pytorch_lightning.loggers import WandbLogger


def train(args):
    # --- DVC 检查逻辑优化 ---
    data_path = "data/processed/train_data.pt"
    if not os.path.exists(data_path):
        print("Warning: Data not found. If this is local, run 'dvc pull'.")
        # 云端时，cloudbuild.yaml 已经帮我们执行过 pull 了

    # W&B 登录
    logger = False
    if args.wandbkey:
        wandb.login(key=args.wandbkey)
        logger = WandbLogger(project="en-zh-translation", name="t5-training-gcp")

    pl.seed_everything(42)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Critical Error: {data_path} missing!")

    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    dataset = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = T5Model(lr=args.lr, batch_size=args.batch_size)

    # 配置 Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        limit_train_batches=0.1 if args.debug_mode else 1.0,
    )

    trainer.fit(model, train_loader)

    # 保存路径优化
    os.makedirs("models", exist_ok=True)
    # 同时保存 state_dict (用于预测) 和 checkpoint (用于恢复)
    torch.save(model.state_dict(), "models/final_model.pt")
    print(" Training Complete! Model saved to models/final_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--wandbkey", default=None, type=str)
    args = parser.parse_args()
    train(args)

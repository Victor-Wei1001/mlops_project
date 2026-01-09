import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from src.models.model import T5Model


def train(args):
    # 1. 种子设置 (Checklist M20: 可重复性)
    pl.seed_everything(42)

    # 2. 加载数据
    data_path = "data/processed/train_data.pt"
    if not os.path.exists(data_path):
        print("Error: train_data.pt not found!")
        return

    data = torch.load(data_path)
    dataset = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])

    # 模板中的 DataLoader 配置
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # 3. 初始化模型 (传入 CLI 参数)
    model = T5Model(lr=args.lr, batch_size=args.batch_size)

    # 4. 训练器设置 (对标模板的各种配置)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        limit_train_batches=0.1 if args.debug_mode else 1.0,  # 模板中的 debug 模式
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )

    # 5. 开始训练
    trainer.fit(model, train_loader)

    # 6. 保存最终权重 (Checklist M17)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/final_model.pt")
    print("Training Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument(
        "--debug_mode", action="store_true", help="Run only 10% of data"
    )

    args = parser.parse_args()
    train(args)

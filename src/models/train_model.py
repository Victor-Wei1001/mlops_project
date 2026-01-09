import argparse
import os
import torch
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader, TensorDataset
from src.models.model import T5Model
from pytorch_lightning.loggers import WandbLogger

def train(args):
    
    if args.wandbkey:
        
        wandb.login(key=args.wandbkey)
        logger = WandbLogger(project="en-zh-translation", name="t5-training")
    else:
       
        logger = False 

    pl.seed_everything(42)

    # 加载数据
    data_path = "data/processed/train_data.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return

    data = torch.load(data_path)
    dataset = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    model = T5Model(lr=args.lr, batch_size=args.batch_size)

    # --- 借鉴点 3: 性能分析 (M12) ---
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,  # 使用上面定义的 logger
        profiler="simple", # 顺便完成 M12 性能分析
        limit_train_batches=0.1 if args.debug_mode else 1.0,
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )

    trainer.fit(model, train_loader)

    # 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/final_model.pt")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--debug_mode", action="store_true")
    # 借鉴点 4: 通过命令行传入 Key
    parser.add_argument("--wandbkey", default=None, type=str, help="W&B API key")

    args = parser.parse_args()
    train(args)
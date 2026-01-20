import logging
import os
import click
import torch
from datasets import load_dataset
from transformers import T5Tokenizer


@click.command()
@click.argument("k", type=int)  # 仿照模板：要求用户输入数据量
@click.option("--output_dir", default="data/processed", help="Output folder")
def main(k, output_dir):
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {k} samples for English-to-Chinese translation")

    # 1. 加载数据
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh", split=f"train[:{k}]")

    # 2. 初始化 Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

    def preprocess(examples):
        inputs = ["translate English to Chinese: " + ex["en"] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        targets = [ex["zh"] for ex in examples["translation"]]
        labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 3. 处理数据
    processed = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    # 4. 强制转换并保存
    os.makedirs(output_dir, exist_ok=True)
    torch_data = {
        "input_ids": torch.tensor(processed["input_ids"]),
        "attention_mask": torch.tensor(processed["attention_mask"]),
        "labels": torch.tensor(processed["labels"]),
    }

    save_path = os.path.join(output_dir, "train_data.pt")
    torch.save(torch_data, save_path)
    logger.info(f"Success! Saved to {save_path} (Size: {os.path.getsize(save_path)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()

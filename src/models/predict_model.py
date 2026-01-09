import torch
import os
from src.models.model import T5Model
from transformers import T5Tokenizer

def predict(text: str):
    model_path = "models/final_model.pt"
    if not os.path.exists(model_path):
        return f"错误：找不到权重文件 {model_path}"

    model = T5Model()
    # 加载我们刚练好的 0.003 Loss 的权重
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

    input_text = "translate English to Chinese: " + text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    print(f"--- 正在翻译: '{text}' ---")
    
    with torch.no_grad():
        generated_ids = model.t5.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True
        )
        
        # --- 这里的缩进非常重要 ---
        print(f"DEBUG: 原始 Token IDs = {generated_ids[0].tolist()}")
        
        translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return translation if translation.strip() else "[模型输出了空结果]"

if __name__ == "__main__":
    test_sentence = "book." 
    result = predict(test_sentence)
    print(f"结果: {result}")
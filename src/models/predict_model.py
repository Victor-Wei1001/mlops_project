import torch
import os
from src.models.model import T5Model
from transformers import T5Tokenizer

def predict(text: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/final_model.pt"
    
    if not os.path.exists(model_path):
        return f"Error: Weight file {model_path} not found."

    model = T5Model().to(device)
    # 加载权重
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    input_text = "translate English to Chinese: " + text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    print(f"--- Translating: '{text}' ---")

    with torch.no_grad():
        generated_ids = model.t5.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True,
        )
        translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return translation

if __name__ == "__main__":
    test_sentence = "This MLOps project is great."
    result = predict(test_sentence)
    print(f"Result: {result}")
'''
Assignment 4
• Train a smaller model (e.g., 1.3B) to replicate this behavior using LoRA or QLoRA.
• Evaluate on small QA tasks.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import copy

# Model identifiers and paths
base_model_id = "deepseek-ai/deepseek-coder-1.3b-base"
lora_model_path = "./lora-deepseek-1.3b-cot"  # Path to your LoRA fine-tuned model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# Load base model (only once)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,  # CPU safe
    device_map="cpu"
)

# Clone base model for student and apply LoRA weights
student_model = copy.deepcopy(base_model)
student_model = PeftModel.from_pretrained(student_model, lora_model_path)

# Inference function
def generate(model, prompt, max_new_tokens=50):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Sample prompts for evaluation
prompts = [
    "Q: A rectangle is 5 meters long and 3 meters wide. What is its area?\nA:",
    "Q: All cats are mammals. Some mammals are dogs. Are all cats dogs?\nA:",
    "Q: If the floor is wet and you slip, what likely happened?\nA:",
    "Q: Who was the first person to walk on the moon?\nA:",
]

# Run and compare
for prompt in prompts:
    print("🟨 Prompt:", prompt.strip())
    print("🔷 Base Model:", generate(base_model, prompt))
    print("🟩 Fine-tuned Model:", generate(student_model, prompt))
    print("=" * 60)

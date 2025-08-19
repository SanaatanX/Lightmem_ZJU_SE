import json
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# 不再需要 PeftModel，因为我们不加载LoRA适配器

# ==============================================================================
# 核心配置：只使用基座模型，并从WikiUpdate中挑选一个问题
# ==============================================================================
BASE_MODEL_PATH = "../Qwen2.5-3B-Instruct"

# 依然使用WikiUpdate的第一个问题作为测试用例
DATA_PATH = "../data/processed/wiki_update_processed.json"
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    first_item = json.load(f)[0]
first_triplet = first_item['ground_truth_triplets'][0]

CONTEXT_TO_TEST = first_item['original_text']
QUESTION_TO_TEST = first_triplet.get('question', first_triplet.get('prompt', '')).format(
    first_triplet.get('subject', ''))
GROUND_TRUTH_ANSWER = first_triplet.get('target', first_triplet.get('answer', ''))


def main():
    print("--- Manual Sanity Check (Testing BASE MODEL ONLY) ---")

    # --- 1. 加载模型和分词器 ---
    # 只加载基座模型，不涉及任何LoRA适配器
    print(f"Loading Base Model from: {BASE_MODEL_PATH}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, quantization_config=quant_config,
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    # --- 2. 构建推理Prompt ---
    instruction = f"Based on the following information, answer the question.\n\nInformation:\n```{CONTEXT_TO_TEST}```\n\nQuestion:\n{QUESTION_TO_TEST}"
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": instruction}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # --- 3. 生成答案并打印结果 ---
    print("\n--- Generating Answer ---")
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=256, do_sample=False)

    response_ids = outputs[0][input_ids.shape[1]:]
    generated_answer = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("\n" + "=" * 50)
    print("QUESTION:")
    print(QUESTION_TO_TEST)
    print("\n" + "-" * 50)
    print("MODEL'S GENERATED ANSWER:")
    print(generated_answer)
    print("\n" + "-" * 50)
    print("GROUND TRUTH ANSWER:")
    print(GROUND_TRUTH_ANSWER)
    print("=" * 50)


if __name__ == '__main__':
    main()

import json  # 用于处理JSON格式的数据，包括加载和保存
import os  # 操作系统模块，用于文件路径操作，例如创建目录
import torch  # PyTorch深度学习框架
import subprocess  # 用于运行外部命令，例如调用评估脚本
from transformers import (  # Hugging Face Transformers库，提供大模型相关工具
    AutoModelForCausalLM,  # 用于自动加载因果语言模型
    AutoTokenizer,  # 用于自动加载与模型对应的分词器
    BitsAndBytesConfig  # 用于配置bitsandbytes库的量化参数
)
from peft import PeftModel  # PEFT (Parameter-Efficient Fine-Tuning) 库，用于加载PEFT模型（在此脚本中未直接使用，但通常用于加载LoRA适配器）
from tqdm import tqdm  # 进度条库，用于可视化循环进度

# 核心配置 (Core Configuration)
# 阿里云DashScope API配置 (用于LongMemEval的LLM裁判，因为LongMemEval的官方评估脚本需要调用LLM)
DASHSCOPE_API_KEY = "sk-ebdb0b9ab9684d4f8c2e6cdaa9048bbc"  # 达摩院API密钥
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 达摩院API的基础URL

# 我们要测试的基座模型路径
BASE_MODEL_PATH = "../Qwen2.5-3B-Instruct"  # Qwen2.5-3B-Instruct模型的本地路径

# 输出评估结果的目录
EVAL_OUTPUT_DIR = "../eval_outputs/"  # 存储所有评估结果的根目录
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)  # 如果目录不存在则创建，exist_ok=True避免目录已存在时报错

EVAL_BATCH_SIZE = 64  # 推理时的批处理大小，一次性处理64个Prompt，以提高效率

# BitsAndBytes量化配置，用于加载4位量化模型
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4位加载
    bnb_4bit_quant_type="nf4",  # 使用NF4量化类型，适用于非对称数据分布
    bnb_4bit_compute_dtype=torch.bfloat16,  # 量化计算时的数据类型，bfloat16有助于保持精度
    bnb_4bit_use_double_quant=True,  # 启用双重量化，进一步降低内存占用
)


# 模型加载与推理函数 (只加载基座模型)
def load_base_model_and_tokenizer():
    """
    只加载基座模型和分词器，并为推理进行编译优化。
    此函数负责模型的初始化，包括量化加载和PyTorch 2.0的编译。

    Returns:
        tuple: 包含加载的模型 (AutoModelForCausalLM) 和分词器 (AutoTokenizer) 的元组。
    """
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    # 加载分词器，trust_remote_code=True 允许加载模型仓库中的自定义代码
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    # 如果pad_token未设置，则将其设置为eos_token，这是Qwen模型常用的设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基座模型，应用量化配置，并指定数据类型和设备映射
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, quantization_config=quantization_config,  # 应用4位量化
        torch_dtype=torch.bfloat16,  # 使用bfloat16数据类型
        device_map="auto",  # 自动将模型层分布到可用设备（如所有GPU）
        trust_remote_code=True,  # 允许加载远程代码
    )
    model.eval()  # 将模型设置为评估模式（关闭dropout等）

    print("Compiling model for faster inference...")
    # 使用PyTorch 2.0的torch.compile进行图编译优化，提高推理速度
    # mode="reduce-overhead" 旨在减少编译开销，fullgraph=True 尝试编译整个图
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model, tokenizer


def generate_batch_answers(model, tokenizer, prompts_batch):
    """
    使用模型为一批prompt生成答案。
    此函数处理批处理推理，并兼容不同分词器输出格式。

    Args:
        model (PreTrainedModel): 已加载的模型实例。
        tokenizer (PreTrainedTokenizer): 已加载的分词器实例。
        prompts_batch (list): 包含多个用户Prompt字符串的列表。

    Returns:
        list: 包含模型为每个Prompt生成的答案字符串列表。
    """
    # 构建消息列表，每个Prompt都包装成Qwen的聊天格式
    messages_batch = [
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}] for prompt
        in prompts_batch]

    tokenizer.padding_side = 'left'  # 设置填充方向为左侧，这对于生成任务通常是推荐的

    # 使用tokenizer的apply_chat_template方法将消息列表转换为token ID张量
    # add_generation_prompt=True: 在最后一条用户消息后添加一个生成提示（例如<|im_start|>assistant\n）
    # return_tensors="pt": 返回PyTorch张量
    # padding=True: 对批次中的序列进行填充，使其长度一致
    # truncation=True: 截断超过max_length的序列
    # max_length=1024: 最大序列长度
    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )

    # 处理不同类型的tokenizer输出：Hugging Face的tokenizer.apply_chat_template可能返回字典或直接张量
    input_ids_tensor = None
    if isinstance(inputs, dict):
        # 标准情况：输出是字典，我们提取'input_ids'键的值
        input_ids_tensor = inputs['input_ids']
    elif isinstance(inputs, torch.Tensor):
        # 特殊情况：输出直接就是input_ids张量
        input_ids_tensor = inputs
    else:
        # 如果遇到其他未知类型，则抛出错误，以便调试
        raise TypeError(f"Unexpected type from tokenizer output: {type(inputs)}. Expected dict or Tensor.")

    # 将输入张量移动到模型所在的设备（GPU）
    input_ids_on_device = input_ids_tensor.to(model.device)

    # 使用模型的generate方法生成答案
    # max_new_tokens=256: 限制生成答案的最大长度
    # do_sample=False: 禁用采样，使用贪婪解码（确定性生成）
    outputs = model.generate(input_ids_on_device, max_new_tokens=256, do_sample=False)

    # 解码逻辑：只解码新生成的token部分
    # outputs[:, input_ids_on_device.shape[1]:] 截取生成结果中除了输入prompt之外的部分
    # skip_special_tokens=True: 跳过特殊token（如<pad>, <s>, </s>等）
    responses = tokenizer.batch_decode(outputs[:, input_ids_on_device.shape[1]:], skip_special_tokens=True)
    return responses


# 主评估逻辑 (Main Evaluation Logic) - 基准
def main():
    """
    主函数：执行基座模型在WikiUpdate和LongMemEval数据集上的基准评估。
    包括加载模型、生成预测、保存预测文件，并自动调用LongMemEval的官方评估脚本。
    """
    print(f"\n{'=' * 20} Running Baseline Evaluation on Base Model {'=' * 20}")

    # 1. 加载唯一的模型：基座模型
    model, tokenizer = load_base_model_and_tokenizer()

    # 2. 在 WikiUpdate 上测试基座模型
    print(f"\n--- Testing baseline on WikiUpdate ---")
    wiki_data_path = "../data/processed/wiki_update_processed.json"  # WikiUpdate数据集路径
    with open(wiki_data_path, 'r', encoding='utf-8') as f:
        wiki_eval_data = json.load(f)  # 加载WikiUpdate评估数据

    wiki_tasks = []  # 存储WikiUpdate的评估任务
    for item in wiki_eval_data:
        # 在基准测试中，上下文永远是未压缩的原文
        context = item['original_text']
        for triplet in item['ground_truth_triplets']:
            raw_question = triplet.get('question', triplet.get('prompt', ''))  # 获取原始问题，兼容不同键名
            # 格式化问题，如果问题包含占位符{}，则用subject填充
            question = raw_question.format(triplet.get('subject', '')) if '{}' in raw_question else raw_question
            # 构建完整的指令Prompt
            instruction = f"Based on the following information, answer the question.\n\nInformation:\n```{context}```\n\nQuestion:\n{question}"
            wiki_tasks.append({
                "model": "baseline_Qwen2.5-3B",  # 模型名称
                "condition": "baseline",  # 评估条件
                "original_id": item['original_id'],  # 原始数据ID
                "question": question,  # 格式化后的问题
                "ground_truth": triplet.get('target', triplet.get('answer', '')),  # 真实答案
                "prompt": instruction  # 用于推理的完整Prompt
            })

    baseline_wiki_predictions = []  # 存储WikiUpdate的预测结果
    # 批处理生成WikiUpdate的答案
    for i in tqdm(range(0, len(wiki_tasks), EVAL_BATCH_SIZE), desc="Generating for WikiUpdate Baseline"):
        batch_tasks = wiki_tasks[i:i + EVAL_BATCH_SIZE]  # 获取当前批次的任务
        prompts_batch = [task['prompt'] for task in batch_tasks]  # 提取批次中的Prompt
        generated_answers = generate_batch_answers(model, tokenizer, prompts_batch)  # 生成答案
        for task, answer in zip(batch_tasks, generated_answers):
            task['generated_answer'] = answer  # 添加生成的答案
            del task['prompt']  # 删除Prompt，因为不再需要
            baseline_wiki_predictions.append(task)

    wiki_output_filename = "eval_wiki_baseline_predictions.json"  # WikiUpdate预测结果文件名
    # 将WikiUpdate的预测结果保存为JSON文件
    with open(os.path.join(EVAL_OUTPUT_DIR, wiki_output_filename), 'w', encoding='utf-8') as f:
        json.dump(baseline_wiki_predictions, f, indent=2, ensure_ascii=False)
    print(f"Baseline WikiUpdate predictions saved to: {wiki_output_filename}")

    # 在 LongMemEval 上测试基座模型
    print(f"\n--- Testing baseline on LongMemEval ---")
    longmem_data_path = "../data/processed/longmem_eval_processed.json"  # LongMemEval数据集路径
    with open(longmem_data_path, 'r', encoding='utf-8') as f:
        longmem_eval_data = json.load(f)  # 加载LongMemEval评估数据

    longmem_tasks = []  # 存储LongMemEval的评估任务
    for item in longmem_eval_data:
        context = item['original_text']  # 上下文同样是原文
        # 构建LongMemEval任务的Prompt
        instruction = f"Here is a conversation history:\n\n```{str(context)}```\n\nBased on the conversation, answer the following question:\n{item['question']}"
        longmem_tasks.append({"question_id": item['original_id'], "prompt": instruction})

    predictions_for_script = []  # 存储LongMemEval的预测结果，格式符合官方评估脚本要求
    # 批处理生成LongMemEval的答案
    for i in tqdm(range(0, len(longmem_tasks), EVAL_BATCH_SIZE), desc="Generating for LongMemEval Baseline"):
        batch_tasks = longmem_tasks[i:i + EVAL_BATCH_SIZE]
        prompts_batch = [task['prompt'] for task in batch_tasks]
        generated_answers = generate_batch_answers(model, tokenizer, prompts_batch)
        for task, answer in zip(batch_tasks, generated_answers):
            # LongMemEval官方脚本需要 'question_id' 和 'hypothesis' 字段
            predictions_for_script.append({"question_id": task['question_id'], "hypothesis": answer})

    pred_filename = "preds_baseline_model.jsonl"  # LongMemEval预测结果文件名
    pred_filepath_abs = os.path.abspath(os.path.join(EVAL_OUTPUT_DIR, pred_filename))  # 预测结果的绝对路径
    # 将LongMemEval的预测结果保存为JSONL格式（每行一个JSON对象）
    with open(pred_filepath_abs, 'w', encoding='utf-8') as f:
        for line in predictions_for_script:
            f.write(json.dumps(line) + '\n')
    print(f"Baseline LongMemEval predictions saved to {pred_filename}")

    # 自动调用官方评估脚本
    print("Running official evaluation script for LongMemEval Baseline...")
    label_filepath_abs = os.path.abspath("../data/longmemeval_local/longmemeval_oracle.json")  # LongMemEval真实标签文件的绝对路径
    eval_script_path = os.path.abspath("../LongMemEval/src/evaluation/evaluate_qa.py")  # 官方评估脚本的绝对路径
    eval_cwd = os.path.dirname(eval_script_path)  # 评估脚本的当前工作目录
    # 构建调用评估脚本的命令
    command = ["python3", os.path.basename(eval_script_path), "qwen-max-aliyun", pred_filepath_abs, label_filepath_abs]

    # 设置环境变量，以便评估脚本能够调用达摩院API
    eval_env = os.environ.copy()  # 复制当前环境变量
    eval_env["OPENAI_API_KEY"] = DASHSCOPE_API_KEY  # 设置API密钥
    eval_env["OPENAI_API_BASE"] = DASHSCOPE_API_BASE  # 设置API基础URL

    # 运行评估脚本，捕获其输出
    result = subprocess.run(command, capture_output=True, text=True, cwd=eval_cwd, env=eval_env)

    if result.returncode == 0:  # 检查脚本是否成功执行
        print("Evaluation script completed successfully.")
        print("--- Official Script Output (Baseline) ---")
        print(result.stdout)  # 打印脚本的标准输出
        print("---------------------------------------")
    else:
        print("Baseline evaluation script failed!")
        print(f"STDERR: {result.stderr}")  # 如果失败，打印标准错误

    print("\n\nBaseline evaluation is complete!")  # 基准评估完成提示


# 确保只有在直接运行脚本时才执行main函数
if __name__ == '__main__':
    main()
import json  # 用于处理JSON格式的数据，包括加载数据集和保存预测结果
import os  # 操作系统模块，用于文件路径操作，如创建目录和拼接路径
import torch  # PyTorch深度学习框架，用于模型操作和GPU管理
import subprocess  # 用于运行外部命令行程序，例如调用LongMemEval的官方评估脚本
from transformers import (  # Hugging Face Transformers库，用于大型语言模型相关操作
    AutoModelForCausalLM,  # 自动加载因果语言模型（例如Qwen2.5-3B-Instruct）
    AutoTokenizer,  # 自动加载与模型对应的分词器
    BitsAndBytesConfig  # 用于配置bitsandbytes库的4位或8位量化加载
)
from peft import PeftModel  # PEFT (Parameter-Efficient Fine-Tuning) 库，用于加载和融合LoRA适配器
from tqdm import tqdm  # 进度条库，用于可视化处理进度，提高用户体验

# 核心配置 (Core Configuration)
# --- 在此处配置阿里云DashScope API信息 ---
# 官方评估脚本需要调用一个LLM来自动打分。
# 在此处填入从阿里云灵积(DashScope)平台获取的API密钥和接入点地址
DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxx"  # **请替换为您的实际DashScope API密钥**
# 阿里云DashScope的接入点地址通常是固定的
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 基座模型路径：这是所有微调模型的基础模型
BASE_MODEL_PATH = "../Qwen2.5-3B-Instruct"
# 训练结果保存目录：微调后的LoRA适配器通常保存在这里
RESULTS_DIR = "../results/"
# 评估输出结果目录：所有生成的预测文件和评估报告将保存到这里
EVAL_OUTPUT_DIR = "../eval_outputs/"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)  # 确保评估输出目录存在，如果不存在则创建

EVAL_BATCH_SIZE = 64  # 推理时每次处理的Prompt数量，可以根据GPU显存调整

# BitsAndBytes量化配置，用于以4位精度加载模型，以节省显存
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4位加载
    bnb_4bit_quant_type="nf4",  # 4位量化的类型，NF4是QLoRA推荐的类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 4位计算时使用bfloat16数据类型，保持较高精度
    bnb_4bit_use_double_quant=True,  # 启用双重量化，进一步压缩内存占用
)


# 模型加载与推理函数 (Model Loading & Inference Functions)
def load_model_and_tokenizer(adapter_path):
    """
    加载基座模型，融合指定的LoRA适配器，并为推理进行编译优化。

    Args:
        adapter_path (str): LoRA适配器在文件系统中的路径。

    Returns:
        tuple: 包含加载并融合后的模型 (AutoModelForCausalLM) 和分词器 (AutoTokenizer) 的元组。
    """
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    # 加载基座模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    # 如果分词器没有pad_token，则将其设置为eos_token，这是Qwen模型常用的做法
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基座模型，应用4位量化配置，并自动映射到可用设备（如GPU）
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    print(f"Loading and merging LoRA adapter from {adapter_path}...")
    # 从指定路径加载LoRA适配器，并将其附加到基座模型上
    model = PeftModel.from_pretrained(model, adapter_path)
    # 将LoRA适配器的权重与基座模型的权重融合，然后卸载LoRA结构。
    # 这样可以得到一个完整的、可直接推理的微调模型，不再需要PEFT库来管理适配器。
    model = model.merge_and_unload()
    model.eval()  # 将模型设置为评估模式，关闭dropout等训练特有的层

    # 为推理过程启用JIT编译，提高推理速度
    print("Compiling model for faster inference...")
    # 'reduce-overhead' 模式非常适合推理任务，因为它旨在减少编译本身带来的开销
    # fullgraph=True 尝试编译整个计算图，以获得最大性能提升
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model, tokenizer


def generate_batch_answers(model, tokenizer, prompts_batch):
    """
    使用模型为一批prompt生成答案。
    该函数处理将Prompt转换为模型输入格式、进行推理以及解码生成结果的全过程。

    Args:
        model (PreTrainedModel): 已经加载并准备好的模型实例。
        tokenizer (PreTrainedTokenizer): 对应的分词器实例。
        prompts_batch (list): 包含多个用户Prompt字符串的列表。

    Returns:
        list: 包含模型为每个Prompt生成的答案字符串列表。
    """
    # 将每个Prompt包装成Qwen模型所需的聊天对话格式（System + User）
    messages_batch = [
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}] for prompt
        in prompts_batch]

    tokenizer.padding_side = 'left'  # 设置填充方向为左侧，这对于因果语言模型的生成任务通常是最佳实践

    # 使用tokenizer的apply_chat_template方法将结构化的消息转换为模型输入ID
    # add_generation_prompt=True: 在用户Prompt后添加一个标记，指示模型开始生成助手回复
    # return_tensors="pt": 返回PyTorch张量
    # padding=True: 对批次中的序列进行填充，使其长度一致
    # truncation=True: 截断超过max_length的序列
    # max_length=1024: 设定的最大输入序列长度
    raw_inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )

    # 检查 tokenizer.apply_chat_template 的输出类型，它可能是字典或直接是张量
    inputs = {}  # 用于存储最终传递给model.generate的字典参数
    if isinstance(raw_inputs, dict):
        # 如果是字典，则遍历其键值对，并将张量类型的value移动到模型设备
        for k, v in raw_inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)  # 将张量移动到GPU
            else:
                inputs[k] = v  # 非张量部分保持不变（例如token_type_ids等，如果存在）
    elif isinstance(raw_inputs, torch.Tensor):
        # 如果直接返回张量（通常意味着只包含input_ids），则将其封装到字典中并移动到设备
        inputs = {'input_ids': raw_inputs.to(model.device)}
        # 注意：在这种情况下，model.generate会默认处理attention_mask。
        # 如果tokenizer在padding=True时未能生成attention_mask，可能需要手动创建。
        # 但通常情况下，padding=True会自动生成并包含在字典输出中。
    else:
        # 如果是其他未知类型，则抛出错误
        raise TypeError(f"Unexpected type for tokenizer output: {type(raw_inputs)}. Expected dict or Tensor.")

    # 使用模型的generate方法生成答案
    # **inputs: 将 inputs 字典作为关键字参数传递给 generate 函数
    # max_new_tokens=256: 限制生成答案的最大长度
    # do_sample=False: 禁用采样，使用贪婪解码（生成最可能的token，结果确定性）
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # 解码生成结果：只解码新生成的token部分
    # outputs[:, inputs['input_ids'].shape[1]:] 截取生成序列中，在原始输入token之后的部分
    # skip_special_tokens=True: 跳过分词器添加的特殊标记（如填充符、句子开始/结束符）
    responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return responses


# 主评估逻辑 (Main Evaluation Logic) - 为批处理模式

def main():
    """
    主函数：执行两个主要评估任务：
    1. 评估 WikiUpdate 压缩系列模型 (实验2b)。
    2. 评估 LongMemEval 压缩系列模型 (使用官方评估脚本和DashScope API)。
    """
    # --- 1. 评估指定的 "run_wiki_compression_" 系列模型 (任务2b) ---
    print(f"\n{'=' * 20} Evaluating WikiUpdate COMPRESSION Models (Task 2b) {'=' * 20}")
    wiki_data_path = "../data/processed/wiki_update_processed.json"  # WikiUpdate数据集路径
    with open(wiki_data_path, 'r', encoding='utf-8') as f:
        wiki_eval_data = json.load(f)  # 加载WikiUpdate评估数据

    # 筛选出所有以 "run_wiki_compression_" 开头的模型目录，并按字母顺序排序
    wiki_model_dirs = sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("run_wiki_compression_")])

    if not wiki_model_dirs:
        print(
            "No 'run_wiki_compression_' models found in the results directory. Skipping WikiUpdate Compression evaluation.")
        return  # 如果没有找到对应的模型目录，则退出

    print(f"Found {len(wiki_model_dirs)} models to evaluate: {wiki_model_dirs}")
    all_wiki_predictions = []  # 列表用于收集所有WikiUpdate模型的预测结果

    for model_dir in wiki_model_dirs:
        adapter_path = os.path.join(RESULTS_DIR, model_dir)  # LoRA适配器的完整路径
        model, tokenizer = load_model_and_tokenizer(adapter_path)  # 加载并融合当前模型的适配器

        # 编写更健壮的逻辑来解析正确的上下文键
        # 模型目录名通常是 "run_wiki_compression_r<rank>_<compression_type>"
        # 例如 "run_wiki_compression_r4_summ_l3_medium"
        dir_parts = model_dir.split('_')
        context_key = "original_text"  # 默认上下文键
        try:
            # 尝试找到以 'r' 开头且后面是数字的部分（表示秩）
            rank_part_index = -1
            for i, part in enumerate(dir_parts):
                if part.startswith('r') and part[1:].isdigit():
                    rank_part_index = i
                    break

            # 如果找到了秩的部分，并且它不是目录名的最后一部分，则秩后面的所有部分拼接起来就是上下文键
            if rank_part_index != -1 and rank_part_index < len(dir_parts) - 1:
                context_key = "_".join(dir_parts[rank_part_index + 1:])
            else:
                # 如果没有找到秩的部分，或者秩是最后一部分（理论上不应该，除非命名不规范）
                raise ValueError("Rank part not found or is the last part.")

        except Exception as e:
            # 如果解析出错，打印警告并使用默认的 'original_text'
            print(
                f"Warning: Could not parse context key from model directory name: '{model_dir}'. Error: {e}. Defaulting to 'original_text'.")

        print(f"Determined context key for {model_dir} is: '{context_key}'")

        tasks = []  # 存储当前模型的评估任务
        for item in wiki_eval_data:
            # 根据解析出的context_key获取上下文，使用.get()方法防止键不存在时报错
            context = item.get(context_key)
            # 如果上下文为空或包含压缩错误标记，则跳过该数据项
            if context is None or "COMPRESSION_ERROR" in str(context): continue

            for triplet in item['ground_truth_triplets']:
                raw_question = triplet.get('question', triplet.get('prompt', ''))  # 获取原始问题，兼容不同键名
                # 格式化问题，如果问题包含占位符{}，则用subject填充
                question = raw_question.format(triplet.get('subject', '')) if '{}' in raw_question else raw_question
                # 构建完整的Prompt指令，包含上下文和问题
                instruction = f"Based on the following information, answer the question.\n\nInformation:\n```{context}```\n\nQuestion:\n{question}"
                tasks.append({
                    "model": model_dir,  # 当前模型的目录名作为模型标识
                    "condition": context_key,  # 当前评估使用的上下文条件
                    "original_id": item['original_id'],  # 原始数据项ID
                    "question": question,  # 格式化后的问题
                    "ground_truth": triplet.get('target', triplet.get('answer', '')),  # 真实答案
                    "prompt": instruction  # 传递给模型的完整Prompt
                })

        if not tasks:
            print(
                f"Warning: No valid tasks were created for {model_dir}. Please check if the context key '{context_key}' exists in your data file.")
            # 释放当前模型和分词器的内存，清空CUDA缓存，为下一个模型做准备
            del model, tokenizer
            torch.cuda.empty_cache()
            continue  # 跳过当前模型，继续下一个

        # 分批次进行推理，并使用tqdm显示进度条
        for i in tqdm(range(0, len(tasks), EVAL_BATCH_SIZE), desc=f"Generating for {model_dir}"):
            batch_tasks = tasks[i:i + EVAL_BATCH_SIZE]  # 获取当前批次的任务
            prompts_batch = [task['prompt'] for task in batch_tasks]  # 提取批次中的所有Prompt
            generated_answers = generate_batch_answers(model, tokenizer, prompts_batch)  # 生成答案
            for task, answer in zip(batch_tasks, generated_answers):
                task['generated_answer'] = answer  # 将生成的答案添加到任务字典中
                del task['prompt']  # 删除Prompt，因为预测完成后不再需要，节省内存
                all_wiki_predictions.append(task)  # 将带生成答案的任务添加到总列表中

        # 释放当前模型和分词器的内存，清空CUDA缓存
        del model, tokenizer
        torch.cuda.empty_cache()

    wiki_output_filename = "eval_wiki_compression_predictions.json"  # WikiUpdate预测结果的输出文件名
    # 将所有WikiUpdate的预测结果保存为JSON文件
    with open(os.path.join(EVAL_OUTPUT_DIR, wiki_output_filename), 'w', encoding='utf-8') as f:
        json.dump(all_wiki_predictions, f, indent=2, ensure_ascii=False)  # indent=2美化输出，ensure_ascii=False支持中文
    print(f"\nAll wiki_compression predictions saved to: {wiki_output_filename}")
    print("Next step: Use `analyze_results.py` to score these predictions.")  # 提示用户后续操作

    print("Skipping WikiUpdate evaluation as it's already complete.")  # 此行可能是调试或旧逻辑残余，保持原样。

    # --- 2. 评估 LongMemEval 系列模型 (官方协议与阿里云API) ---
    print(f"\n{'=' * 20} Evaluating LongMemEval Models (Official Protocol with Alibaba Cloud API) {'=' * 20}")
    longmem_data_path = "../data/processed/longmem_eval_processed.json"  # LongMemEval数据集路径
    with open(longmem_data_path, 'r', encoding='utf-8') as f:
        longmem_eval_data = json.load(f)  # 加载LongMemEval评估数据

    # 筛选出所有以 "run_longmem_" 开头的模型目录
    longmem_model_dirs = sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("run_longmem_")])
    all_longmem_scores = {}  # 字典用于存储所有LongMemEval模型的最终分数

    if not longmem_model_dirs:
        print("No 'run_longmem_' models found in the results directory. Skipping LongMemEval evaluation.")
        return  # 如果没有找到对应的模型目录，则退出

    for model_dir in longmem_model_dirs:
        adapter_path = os.path.join(RESULTS_DIR, model_dir)  # 当前模型的LoRA适配器路径
        model, tokenizer = load_model_and_tokenizer(adapter_path)  # 加载并融合当前模型的适配器

        # 从模型目录名中解析出压缩类型（例如 "summ_l3_medium"）
        # 模型目录名格式通常是 "run_longmem_<compression_type>"
        comp_type = "_".join(model_dir.split("_")[2:])  # 截取目录名中表示压缩类型的部分

        tasks = []  # 存储当前模型的LongMemEval任务
        for item in longmem_eval_data:
            # 根据解析出的压缩类型获取上下文
            context = item.get(comp_type)
            if context is None or "COMPRESSION_ERROR" in str(context): continue  # 跳过无效或压缩失败的数据

            # 构建LongMemEval任务的Prompt指令
            # 注意这里将上下文转换为字符串，以防它不是纯字符串类型
            instruction = f"Here is a conversation history:\n\n```{str(context)}```\n\nBased on the conversation, answer the following question:\n{item['question']}"
            tasks.append({"question_id": item['original_id'], "prompt": instruction})

        # a. 生成预测结果并保存为官方要求的 .jsonl 格式
        # LongMemEval官方评估脚本需要JSON Lines (.jsonl) 格式的预测文件
        pred_filename = f"preds_{model_dir}.jsonl"  # 预测结果文件名
        pred_filepath_abs = os.path.abspath(os.path.join(EVAL_OUTPUT_DIR, pred_filename))  # 预测文件的绝对路径

        with open(pred_filepath_abs, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(tasks), EVAL_BATCH_SIZE), desc=f"Generating for {model_dir}"):
                batch_tasks = tasks[i:i + EVAL_BATCH_SIZE]
                prompts_batch = [task['prompt'] for task in batch_tasks]
                generated_answers = generate_batch_answers(model, tokenizer, prompts_batch)
                for task, answer in zip(batch_tasks, generated_answers):
                    # 将生成的答案格式化为符合LongMemEval脚本要求的JSON对象
                    result_line = {"question_id": task['question_id'], "hypothesis": answer}
                    f.write(json.dumps(result_line) + '\n')  # 写入一行JSON
        print(f"Predictions saved to {pred_filepath_abs}")

        # b. 使用官方文档指定的正确命令，自动调用评估脚本
        print(f"Running official evaluation script for {model_dir}...")
        label_filepath_abs = os.path.abspath("../data/longmemeval_local/longmemeval_oracle.json")  # 真实标签文件的绝对路径
        eval_script_path = os.path.abspath("../LongMemEval/src/evaluation/evaluate_qa.py")  # 官方评估脚本的绝对路径
        eval_cwd = os.path.dirname(eval_script_path)  # 评估脚本的执行目录

        # 构建调用评估脚本的命令。
        # "qwen-max-aliyun" 是官方脚本内部用来识别并调用DashScope API的字符串。
        command = [
            "python3", os.path.basename(eval_script_path),
            "qwen-max-aliyun",
            pred_filepath_abs,
            label_filepath_abs
        ]

        # 准备一个“定制”的执行环境，将阿里云配置“伪装”成OpenAI的配置，
        # 因为LongMemEval的评估脚本可能内部使用了OpenAI兼容的客户端。
        eval_env = os.environ.copy()  # 复制当前进程的环境变量
        eval_env["OPENAI_API_KEY"] = DASHSCOPE_API_KEY  # 设置DashScope API密钥到OpenAI兼容的环境变量
        eval_env["OPENAI_API_BASE"] = DASHSCOPE_API_BASE  # 设置DashScope API地址到OpenAI兼容的环境变量

        # 运行评估脚本，capture_output=True 捕获标准输出和标准错误，text=True 以文本模式处理输出
        result = subprocess.run(command, capture_output=True, text=True, cwd=eval_cwd, env=eval_env)

        # c. 解析并保存分数
        if result.returncode == 0:  # 检查子进程的返回码，0表示成功
            print("Evaluation script completed successfully.")
            print("--- Official Script Output ---")
            print(result.stdout)  # 打印评估脚本的标准输出
            print("----------------------------")
            # 将评估结果（包括成功状态和脚本输出）存储起来
            all_longmem_scores[model_dir] = {"status": "success", "output": result.stdout}
        else:
            print(f"Evaluation script failed for {model_dir}!")
            print("--- STDERR ---")
            print(result.stderr)  # 打印脚本的标准错误
            print("--- STDOUT ---")
            print(result.stdout)  # 打印脚本的标准输出
            # 记录失败状态和错误信息
            all_longmem_scores[model_dir] = {"status": "failed", "error": result.stderr}

        # 释放当前模型和分词器的内存，清空CUDA缓存，为下一个模型做准备
        del model, tokenizer
        torch.cuda.empty_cache()

    # 保存所有LongMemEval的评估分数到一个JSON文件
    longmem_output_filename = "eval_longmem_all_scores.json"
    with open(os.path.join(EVAL_OUTPUT_DIR, longmem_output_filename), 'w', encoding='utf-8') as f:
        json.dump(all_longmem_scores, f, indent=2)  # indent=2美化输出
    print(f"All LongMemEval scores saved to: {longmem_output_filename}")

    print("\n\nAll automated evaluation tasks are complete!")
    print("Next step: Manual or LLM-as-a-Judge scoring for WikiUpdate results.")  # 提示后续步骤


# 确保只有在直接运行脚本时才执行main函数
if __name__ == '__main__':
    main()
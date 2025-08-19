# ==============================================================================
# 导入所有必需的库
# ==============================================================================
import json
import os
import time
import random  # 引入random库，用于在API重试时增加“抖动”，避免请求冲突
from datasets import Dataset
from openai import OpenAI, RateLimitError  # 显式导入OpenAI客户端和特定的错误类型，用于智能重试
from concurrent.futures import ThreadPoolExecutor # 导入Python现代的、高效的并行处理工具
from tqdm import tqdm  # 导入tqdm，用于在循环中显示美观、直观的进度条
from functools import partial # 导入partial，用于在并行化时“冻结”函数的某些参数

# ==============================================================================
# 核心配置：定义所有实验的“常量”
# ==============================================================================
# 1. API客户端配置
# 请在此处填入您从阿里云灵积(DashScope)平台获取的API密钥
SILICONFLOW_API_KEY = "sk-xxxxxxxxxx"

# 2. 初始化API客户端
# 我们使用OpenAI的SDK，但通过指定base_url，让它将请求发送到阿里云的服务器
client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 3. 定义所有压缩策略的指令(Prompt)模板
# 这是一个字典，键(key)是压缩级别的名称，值(value)是发送给大模型的具体指令
PROMPTS = {
    "summ_l1_slight": "Please slightly condense the following text, retaining almost all details but improving conciseness. Only return the condensed text, without any extra explanations.\n\nText:\n```{text}```",
    "summ_l2_light": "Please summarize the following text into a detailed summary of about 100 words, preserving key entities and events. Only return the summary, without any extra explanations.\n\nText:\n```{text}```",
    "summ_l3_medium": "Please summarize the following text into a concise summary of about 50 words. Focus on the main points. Only return the summary, without any extra explanations.\n\nText:\n```{text}```",
    "summ_l4_heavy": "Please condense the following text into one or two core sentences, retaining only the most critical information. Only return the summary, without any extra explanations.\n\nText:\n```{text}```",
    "summ_l5_extreme": "Please extract only the absolute most essential fact or conclusion from the following text, expressed in a single short phrase. Only return the phrase, without any extra explanations.\n\nText:\n```{text}```",
    "ext": "Please extract all core entities, their relationships, and key events from the following text. Return the result in strict JSON format, without any code block markers or extra explanations. The JSON format should be: {{\"entities\": [\"Entity 1\", \"Entity 2\"], \"relations\": [{{\"subject\": \"Entity A\", \"relation\": \"Relationship\", \"object\": \"Entity B\"}}], \"events\": [\"Event 1 description\", \"Event 2 description\"]}}.\n\nText:\n```{text}```"
}


# ==============================================================================
# 核心工作函数
# ==============================================================================

def compress_text(text: str, prompt_template: str) -> str:
    """
    通过API调用大模型，对单个文本进行压缩。
    这个函数是整个数据处理流程中最核心的“工人”。
    它还内置了“无限智能重试”机制，以确保在面对API速率限制时程序的稳定性。
    """
    # 设定一个输入长度上限，防止因文本过长而导致API直接报错
    max_input_length = 8000
    if len(str(text)) > max_input_length: # 确保将输入转为字符串再判断长度
        text = str(text)[:max_input_length]

    # --- 智能重试机制的参数 ---
    attempt = 0  # 当前重试次数计数器
    base_wait_time = 2  # 初始等待时间为2秒
    max_wait_time = 60  # 设置最长单次等待时间为60秒，防止因指数增长导致永久等待

    while True:  # 使用无限循环，直到成功或遇到不可恢复的错误
        try:
            # 发送API请求
            response = client.chat.completions.create(
                model="qwen2.5-7b-instruct-1m", # 您选择的压缩模型
                messages=[
                    {"role": "system", "content": "You are an expert in text processing. Please follow the user's instructions carefully and provide only the required output without any extra explanations."},
                    {"role": "user", "content": prompt_template.format(text=text)}
                ],
                temperature=0.0,    # 使用0温度，确保输出的确定性
                max_tokens=1024,    # 设置一个合理的输出长度上限
                stream=False
            )
            # 如果API调用成功，则清理并返回结果，同时跳出无限循环
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            # 专门捕获“速率限制”错误
            # 计算下一次重试的等待时间，采用“指数退避+随机抖动”策略
            wait_time = min(max_wait_time, base_wait_time * (2 ** attempt)) + random.uniform(0, 1)
            print(f"    - Rate limit reached. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1})")
            time.sleep(wait_time) # 等待指定时间
            attempt += 1 # 增加尝试次数，以便下次计算更长的等待时间

        except Exception as e:
            # 捕获其他所有未知错误（如网络中断、API服务内部错误等）
            # 对于这些错误，我们不应该无限重试，只尝试2次
            if attempt < 2:
                print(f"    - An unexpected error occurred: {e}. Retrying in 5 seconds... (Attempt {attempt + 1}/2)")
                time.sleep(5)
                attempt += 1
                continue # 继续下一次循环尝试
            else:
                # 如果重试2次后依然失败，则彻底放弃，并返回一个明确的错误标识
                print(f"    - An unexpected error occurred and retries failed: {e}")
                return f"COMPRESSION_ERROR: {e}"


def process_single_item(item, item_type='wiki'):
    """
    处理单个数据条目(item)的完整工作流。
    这个函数将被并行地应用到每一个数据条目上。
    """
    # --- 第一步：根据数据类型，提取原始文本和元数据 ---
    if item_type == 'wiki':
        # 如果是WikiUpdate数据，需要从嵌套结构中提取信息
        nested_data = item['requested_rewrite']
        original_text = nested_data['fact_new_uns']
        # 将原始ID和用于最终评估的“标准答案”也一并打包
        processed_item = {
            'original_id': item['case_id'],
            'original_text': original_text,
            'ground_truth_triplets': nested_data['unsfact_triplets_GPT']
        }
    else:  # longmem
        # 如果是LongMemEval数据，结构更扁平
        original_text = item['haystack_sessions']
        # 打包ID、原文、以及用于最终评估的问题和答案
        processed_item = {
            'original_id': item['question_id'],
            'original_text': original_text,
            'question': item['question'],
            'answer': item['answer']
        }

    # --- 第二步：循环调用API，生成所有压缩版本 ---
    # 遍历我们预先定义好的PROMPTS字典
    for key, prompt in PROMPTS.items():
        # 调用核心压缩函数，并将返回的压缩文本，以相应的键名存入我们正在构建的字典中
        processed_item[key] = compress_text(original_text, prompt)

    # --- 第三步：返回包含所有版本信息的完整数据对象 ---
    return processed_item


def main():
    """
    主执行函数，负责调度整个数据处理流程。
    它采用并行处理，并能分别处理两个不同的数据集。
    """
    # --- 准备工作：创建输出目录 ---
    processed_dir = '../data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    # --- 为两个数据集分别设置并行处理的“工人”数量 ---
    MAX_WORKERS1 = 32 # 用于处理WikiUpdate
    MAX_WORKERS2 = 32 # 用于处理LongMemEval

    # ============================================================
    # --- 流程一：并行处理 WikiUpdate 数据集 ---
    # ============================================================
    print("\n--- Processing WikiUpdate Dataset (in parallel, order guaranteed) ---")
    # 1. 加载原始数据，并只取前500条
    with open('../data/WikiUpdate.json', 'r', encoding='utf-8') as f:
        wiki_update_raw = json.load(f)[:500]

    # 2. 使用partial“冻结”process_single_item函数的item_type参数，使其适用于并行化
    process_wiki_item_partial = partial(process_single_item, item_type='wiki')

    # 3. 创建线程池，开始并行处理
    processed_wiki_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS1) as executor:
        # executor.map会并行地将process_wiki_item_partial函数应用到wiki_update_raw列表的每个元素上
        # 它会保证返回结果的顺序与输入顺序完全一致
        results_iterator = executor.map(process_wiki_item_partial, wiki_update_raw)

        # 4. 使用tqdm显示进度条，并收集所有处理完成的结果
        for result in tqdm(results_iterator, total=len(wiki_update_raw), desc="Processing WikiUpdate"):
            processed_wiki_data.append(result)

    # 5. 将包含所有压缩版本的完整数据，保存到新的JSON文件中
    wiki_output_path = os.path.join(processed_dir, 'wiki_update_processed.json')
    with open(wiki_output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_wiki_data, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved processed WikiUpdate data to: {wiki_output_path}")

    # ============================================================
    # --- 流程二：并行处理 LongMemEval 数据集 ---
    # ============================================================
    print("\n--- Processing LongMemEval Dataset (in parallel, order guaranteed) ---")

    # 1. 从本地文件加载数据，并进行预清洗
    print("Pre-loading and cleaning LongMemEval data...")
    with open('../data/longmemeval_local/longmemeval_oracle.json', 'r', encoding='utf-8') as f:
        longmem_data_raw = json.load(f)

    # 清洗数据：确保'answer'字段总是字符串类型，以避免后续处理出错
    for item in longmem_data_raw:
        item['answer'] = str(item['answer'])

    # 2. 从清洗后的内存列表创建Dataset对象
    longmem_dataset = Dataset.from_list(longmem_data_raw)
    print("Cleaning complete. Starting parallel processing...")

    # 3. 再次使用partial“冻结”参数
    process_longmem_item_partial = partial(process_single_item, item_type='longmem')

    # 4. 创建线程池，开始并行处理
    processed_longmem_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS2) as executor:
        results_iterator = executor.map(process_longmem_item_partial, longmem_dataset)
        for result in tqdm(results_iterator, total=len(longmem_dataset), desc="Processing LongMemEval"):
            processed_longmem_data.append(result)

    # 5. 将处理好的数据保存到新的JSON文件中
    longmem_output_path = os.path.join(processed_dir, 'longmem_eval_processed.json')
    with open(longmem_output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_longmem_data, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved processed LongMemEval data to: {longmem_output_path}")


# 确保这个脚本是作为主程序运行时，才执行main函数
if __name__ == '__main__':
    main()
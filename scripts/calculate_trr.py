import json # 用于处理JSON格式的数据，例如加载数据集
from transformers import AutoTokenizer # Hugging Face Transformers库，用于加载预训练的分词器
import numpy as np # 用于数值计算，特别是计算平均值

def calculate_trr(original_text, compressed_text, tokenizer):
    """
    计算单条文本的Token缩减率(TRR)。
    TRR衡量了压缩文本相对于原始文本在token数量上的减少比例。
    增加了对不同输入类型的健壮性检查，确保输入是字符串类型。

    Args:
        original_text (str or list): 原始文本，可能是一个字符串或字符串列表。
        compressed_text (str): 压缩后的文本。
        tokenizer: 用于将文本转换为token的分词器实例。

    Returns:
        float or None: 计算出的Token缩减率（0到1之间的浮点数）。
                       如果原始token数量为0，返回0。
                       如果输入无效（如compressed_text不是字符串或包含错误标记），返回None。
    """

    # ↓↓↓ 核心修正：如果original_text不是字符串（例如是List），先将其转换为字符串 ↓↓↓
    # 这一步确保了无论original_text的初始类型如何，都能被正确地分词。
    if not isinstance(original_text, str):
        original_text = str(original_text) # 将非字符串类型的original_text转换为字符串

    # 检查压缩文本是否为有效字符串
    if not isinstance(compressed_text, str):
        print(f"Warning: compressed_text is not a string: {type(compressed_text)}. Skipping.")
        return None # 如果压缩文本不是字符串，则无法计算，返回None

    # 检查压缩是否失败：如果压缩文本中包含特定错误标记，则认为压缩失败
    if "COMPRESSION_ERROR" in compressed_text:
        print(f"Warning: 'COMPRESSION_ERROR' found in compressed_text. Skipping.")
        return None # 压缩失败，返回None

    # 使用分词器计算原始文本和压缩文本的token数量
    # tokenizer.encode() 方法将文本转换为token ID列表，然后通过len()获取token数量
    original_tokens = len(tokenizer.encode(original_text))
    compressed_tokens = len(tokenizer.encode(compressed_text))

    # 避免除以零的错误：如果原始token数量为0，则缩减率为0 (因为没有内容可以缩减)
    if original_tokens == 0:
        return 0.0 # 返回浮点数0.0

    # 计算Token缩减率：1 - (压缩后token数 / 原始token数)
    return 1 - (compressed_tokens / original_tokens)


def main():
    """
    主函数：加载指定的数据集，遍历其中不同压缩级别的文本，
    计算每种压缩类型的平均Token缩减率(TRR)，并打印结果。
    """
    print("Loading Qwen Tokenizer...")
    # 加载Qwen2.5-3B-Instruct模型的分词器。
    # trust_remote_code=True 允许加载模型仓库中的自定义代码（如果需要）。
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)

    # 定义需要分析的文件路径，键是数据集名称，值是对应的JSON文件路径
    files_to_analyze = {
        "WikiUpdate": "../data/processed/wiki_update_processed.json",
        "LongMemEval": "../data/processed/longmem_eval_processed.json"
    }

    # 定义需要计算TRR的压缩类型（对应JSON文件中的键名）
    compression_types = [
        'summ_l1_slight', # 轻微摘要
        'summ_l2_light',  # 少量摘要
        'summ_l3_medium', # 中等摘要
        'summ_l4_heavy',  # 大量摘要
        'summ_l5_extreme',# 极端摘要
        'ext'             # 可能是某种提取式压缩
    ]

    # 遍历每个需要分析的数据集
    for name, path in files_to_analyze.items():
        print(f"\n--- Analyzing Compression for: {name} ---") # 打印当前正在分析的数据集名称
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f) # 加载JSON格式的数据
        except FileNotFoundError:
            # 如果文件未找到，打印错误信息并跳过当前数据集
            print(f"Error: File not found at {path}. Please make sure Step 2 was completed successfully.")
            continue # 继续处理下一个数据集（如果有的话）

        # 遍历每种压缩类型，计算其平均TRR
        for comp_type in compression_types:
            all_trr_values = [] # 列表用于存储当前压缩类型的所有有效TRR值
            for item in data: # 遍历数据集中的每个数据项
                # 调用calculate_trr函数计算单条数据的TRR
                # item['original_text'] 是原始文本
                # item.get(comp_type) 用于安全地获取压缩文本，如果键不存在则返回None
                trr = calculate_trr(item['original_text'], item.get(comp_type), tokenizer)
                if trr is not None: # 只有当TRR计算成功（不为None）时才添加到列表中
                    all_trr_values.append(trr)

            # 如果收集到了有效的TRR值，则计算并打印平均值
            if all_trr_values:
                average_trr = np.mean(all_trr_values) # 使用numpy计算平均值
                # 打印平均TRR，并格式化为百分比形式，保留两位小数
                print(f"Average TRR for '{comp_type}': {average_trr:.2%}")
            else:
                # 如果没有有效数据用于计算TRR，打印提示信息
                print(f"No valid data found for '{comp_type}' to calculate TRR.")


# 确保只有在直接运行脚本时才执行main函数
if __name__ == '__main__':
    main()
import json  # 用于处理JSON格式的数据，包括加载和保存
import os  # 操作系统模块，用于文件路径操作 (在此脚本中主要用于构建路径字符串)
import time  # 时间模块，用于在需要时暂停执行 (在backoff装饰器中隐式使用)
from collections import defaultdict  # 提供了带默认值的字典，方便分组统计
import numpy as np  # 用于数值计算，特别是计算平均值
from tqdm import tqdm  # 进度条库，用于可视化并行处理的进度
from concurrent.futures import ThreadPoolExecutor  # 用于创建线程池，实现并行执行任务
import backoff  # 用于实现重试逻辑，处理可能出现的API错误或限速
from openai import OpenAI, RateLimitError, APIError  # OpenAI API客户端库，RateLimitError和APIError用于错误处理

# 核心配置 (无变化)
DASHSCOPE_API_KEY = "sk-ebdb0b9ab9684d4f8c2e6cdaa9048bbc"  # 达摩院API密钥，用于认证
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 达摩院API的基础URL
JUDGE_MODEL_ID = "qwen-plus"  # 评估模型（裁判LLM）的ID，这里使用Qwen-Plus
MAX_WORKERS = 16  # 并行处理的最大线程数，影响同时进行的API请求数量

# 初始化OpenAI客户端，指向达摩院API
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_API_BASE,
)


# 评分函数 (无变化)
@backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=5)
def get_llm_score(ground_truth: str, generated_answer: str) -> int:
    """
    调用LLM裁判（Qwen-Plus）对生成的答案进行打分。
    使用@backoff装饰器处理RateLimitError和APIError，最多重试5次，并采用指数退避策略。

    Args:
        ground_truth (str): 真实答案，作为评估的基准。
        generated_answer (str): 模型生成的答案。

    Returns:
        int: LLM裁判给出的分数 (1-5)。如果API调用失败或返回非预期值，则返回0。
    """
    # 构建发送给LLM裁判的Prompt
    prompt = f"""You are an objective evaluator. Based on the Ground Truth, please evaluate the Generated Answer on a scale of 1 to 5, where 1 means completely wrong and 5 means perfectly correct and relevant. Your evaluation should be strict. If the generated answer is not directly and fully supported by the ground truth, you should deduct points. Return ONLY the integer score (1, 2, 3, 4, or 5) and nothing else.

Ground Truth: "{ground_truth}"
Generated Answer: "{generated_answer}"

Your Score:"""
    try:
        # 调用OpenAI API的聊天完成端点
        response = client.chat.completions.create(
            model=JUDGE_MODEL_ID,  # 指定裁判模型
            messages=[{"role": "user", "content": prompt}],  # 传入用户Prompt
            temperature=0,  # 温度设置为0，使LLM的输出确定性更高（更客观）
            max_tokens=5  # 限制生成最大token数，预期只返回一个数字
        )
        score_text = response.choices[0].message.content.strip()  # 提取LLM生成的文本并去除空白
        if score_text in ['1', '2', '3', '4', '5']:  # 检查返回的文本是否是有效分数
            return int(score_text)  # 转换为整数并返回
        else:
            # 如果LLM返回了非预期值，打印警告并返回0
            print(f"Warning: LLM returned a non-score value: '{score_text}' for answer '{generated_answer[:50]}...'")
            return 0
    except Exception as e:
        # 捕获API调用过程中可能发生的其他异常，打印错误信息并返回0
        print(f"LLM scoring API call failed: {e}")
        return 0


def score_single_prediction(prediction_item: dict) -> dict:
    """
    为单个预测条目获取分数并将其添加到字典中。
    这是一个包装函数，方便在ThreadPoolExecutor中使用。

    Args:
        prediction_item (dict): 包含 'ground_truth' 和 'generated_answer' 的字典。

    Returns:
        dict: 原始 prediction_item 字典，新增 'score' 字段。
    """
    score = get_llm_score(prediction_item['ground_truth'], prediction_item['generated_answer'])
    prediction_item['score'] = score  # 将获取到的分数添加到字典中
    return prediction_item


# 主分析逻辑
def main():
    """
    主函数：执行整个评估流程。
    加载所有预测数据，并行调用LLM裁判进行评分，保存带有分数的预测，
    最后按模型名称分组计算并展示平均分数。
    """
    # 我们将评测目标指向包含所有模型预测的总文件
    eval_file_path = "../eval_outputs/eval_wiki_compression_predictions.json"  # 待评估的预测文件路径
    print(f"Loading all predictions from {eval_file_path}...")
    try:
        with open(eval_file_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)  # 加载预测数据
    except FileNotFoundError:
        print(f"Error: File not found at {eval_file_path}.")  # 文件不存在则报错并退出
        return

    print(f"Scoring all {len(predictions)} predictions in parallel (max_workers={MAX_WORKERS})...")

    scored_predictions = []  # 列表用于存储所有带分数的预测结果
    # 使用ThreadPoolExecutor进行并行评分
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # map函数将score_single_prediction函数应用于predictions列表中的每个元素
        # tqdm用于显示进度条
        results_iterator = executor.map(score_single_prediction, predictions)
        for result in tqdm(results_iterator, total=len(predictions), desc="Scoring All WikiUpdate Predictions"):
            scored_predictions.append(result)

    # 保存带有分数的结果到新的JSON文件
    scored_output_path = "../eval_outputs/eval_wiki_compression_predictions_scored.json"
    with open(scored_output_path, 'w', encoding='utf-8') as f:
        json.dump(scored_predictions, f, indent=2, ensure_ascii=False)  # indent=2用于美化输出，ensure_ascii=False支持中文
    print(f"\nScored predictions have been saved to {scored_output_path}")

    # 使用 "model" 字段作为分组的唯一键，统计每个模型的得分
    scores_by_model = defaultdict(list)  # 使用defaultdict，默认值为列表
    for pred in scored_predictions:
        # 不再使用有歧义的'condition'，而是用唯一的模型名'model'字段进行分组
        # 并且只统计得分大于0（即有效得分）的预测
        if pred.get('score', 0) > 0:
            scores_by_model[pred['model']].append(pred['score'])  # 将分数添加到对应模型的列表中

    # 计算每个模型的平均分并打印结果
    final_results = {}  # 存储最终聚合结果的字典
    print("\n" + "=" * 70)
    print("--- Final Aggregated Evaluation Results (Grouped by Unique Model) ---")
    print("=" * 70)

    # 为了报告的清晰和可重复性，按模型名称的字母顺序打印结果
    sorted_models = sorted(scores_by_model.keys())
    for model_name in sorted_models:
        scores = scores_by_model[model_name]
        if scores:  # 确保有分数可以计算
            average_score = np.mean(scores)  # 计算平均分
            final_results[model_name] = {  # 将结果存储到final_results字典
                "average_score": round(average_score, 4),  # 保留四位小数
                "count": len(scores)  # 记录评估的样本数量
            }
            # 格式化打印每个模型的平均分和样本数量
            print(f"Model: {model_name:<60} | Avg Score: {average_score:.4f} | Count: {len(scores)}")

    # 保存最终的、聚合后的分数到JSON文件
    final_results_path = "../eval_outputs/final_all_models_aggregated_scores.json"
    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)  # indent=2美化输出
    print(f"\nAggregated results for ALL models saved to {final_results_path}")
    print("\nFinal analysis is fully complete!")  # 脚本执行完毕的提示


# 确保只有在直接运行脚本时才执行main函数
if __name__ == '__main__':
    main()
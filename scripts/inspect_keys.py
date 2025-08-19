import json

# 这个脚本会打印出LongMemEval文件中第一个数据条目的所有标签（keys）
try:
    # ↓↓↓ 关键改动：指向longmemeval的文件 ↓↓↓
    with open('../data/longmemeval_local/longmemeval_oracle.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        if data and isinstance(data, list) and len(data) > 0:
            print("LongMemEval文件中第一个数据条目包含的标签如下：")
            print(list(data[0].keys()))
        else:
            print("文件为空或格式不正确。")
except Exception as e:
    print(f"读取文件时发生错误: {e}")
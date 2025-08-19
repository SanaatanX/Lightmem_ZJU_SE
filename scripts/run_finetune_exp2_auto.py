import json  # 用于处理JSON格式的数据
import os  # 用于操作系统相关功能，例如路径操作 (在此脚本中未直接使用，但通常用于目录创建等)
from functools import partial  # 用于创建偏函数，方便在map函数中传递额外的参数

import torch  # PyTorch深度学习框架
from datasets import Dataset  # Hugging Face Datasets库中的Dataset类，用于构建数据集
from transformers import (  # Hugging Face Transformers库，用于大模型相关操作
    AutoModelForCausalLM,  # 自动加载因果语言模型 (如Qwen2)
    AutoTokenizer,  # 自动加载对应的分词器
    TrainingArguments,  # 训练参数配置类
    Trainer,  # 训练器类，用于简化模型训练过程
    DataCollatorForSeq2Seq,  # 序列到序列数据整理器，用于批处理和填充
    BitsAndBytesConfig  # 用于配置bitsandbytes库的4位或8位量化
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # PEFT (Parameter-Efficient Fine-Tuning) 库


# LoraConfig: LoRA配置类
# get_peft_model: 将LoRA适配器添加到模型
# prepare_model_for_kbit_training: 为量化训练准备模型


# 1. 数据格式化函数 (此函数与之前的脚本完全相同)

def format_data_for_qwen(example, tokenizer):
    """
    将单个数据条目格式化为Qwen2模型接受的指令格式。
    Qwen2模型通常使用特定的聊天模板来处理指令和对话。

    Args:
        example (dict): 包含 'context', 'question', 'answer' 字段的字典。
        tokenizer: 用于对消息进行分词的tokenizer实例。

    Returns:
        dict: 包含 'input_ids' 和 'labels' 的字典，用于模型训练。
              'input_ids' 是分词后的输入序列ID。
              'labels' 是用于计算损失的目标序列ID (在此处与input_ids相同，因为是自回归任务)。
    """
    context = example['context']  # 从输入example中提取上下文信息
    question = example['question']  # 从输入example中提取问题
    answer = example['answer']  # 从输入example中提取答案 (即模型期望的输出)
    system_prompt = "You are a helpful assistant that answers questions based on the provided context."  # 设定系统角色提示
    # 构造用户指令，包含信息和问题，采用Markdown代码块格式化信息
    instruction = f"Based on the following information, answer the question.\n\nInformation:\n```{context}```\n\nQuestion:\n{question}"

    # 按照Qwen2的聊天模板构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},  # 系统消息
        {"role": "user", "content": instruction},  # 用户消息
        {"role": "assistant", "content": answer}  # 助手消息 (即真实答案)
    ]

    # 使用tokenizer的apply_chat_template方法将消息列表转换为token ID
    tokenized_example = tokenizer.apply_chat_template(
        messages,
        tokenize=True,  # 执行分词操作
        add_generation_prompt=False,  # 不添加生成提示 (因为是训练数据，不是用于生成)
        padding='max_length',  # 填充到最大长度
        truncation=True,  # 截断超过最大长度的序列
        max_length=1024  # 设定的最大序列长度
    )
    # 对于因果语言模型训练，通常将input_ids作为labels，模型会学习预测下一个token
    return {"input_ids": tokenized_example, "labels": tokenized_example.copy()}


# 2. 实验2a与2b的全自动主训练逻辑
def main():
    """
    此脚本专门用于全自动执行实验2a和2b的高效精简版。
    实验2a：容量分析 (Capacity Analysis) - 探讨不同训练数据量对模型性能的影响。
    实验2b：压缩分析 (Compression Analysis) - 探讨不同上下文压缩级别对模型性能的影响。
    """
    # --- 通用配置：模型、量化、训练参数 ---
    model_id = "../Qwen2.5-3B-Instruct"  # 预训练模型的本地路径或Hugging Face模型ID
    quantization_config = BitsAndBytesConfig(  # 量化配置，用于4位量化训练 (QLoRA)
        load_in_4bit=True,  # 加载4位量化模型
        bnb_4bit_quant_type="nf4",  # 4位量化的类型，nf4是QLoRA推荐的类型
        bnb_4bit_compute_dtype=torch.bfloat16,  # 量化计算的数据类型，bf16可以保持较高精度
        bnb_4bit_use_double_quant=True  # 使用嵌套量化，进一步节省内存
    )
    training_args = TrainingArguments(  # 训练参数配置
        per_device_train_batch_size=48,  # 每个设备的训练批次大小 (为H800 GPU优化)
        gradient_accumulation_steps=1,  # 梯度累积步数，1表示不累积
        num_train_epochs=3,  # 训练的总epoch数量
        learning_rate=2e-4,  # 学习率
        bf16=True,  # 启用bfloat16混合精度训练
        logging_steps=20,  # 每隔20步记录一次日志
        save_strategy="epoch",  # 每个epoch结束时保存模型
        save_total_limit=1,  # 最多保存一个检查点 (最近的)
        report_to="none",  # 不向任何报告工具报告 (如wandb、tensorboard)
        torch_compile=True,  # 启用PyTorch 2.0的torch.compile进行图编译优化
    )

    # --- 定义所有正交实验的变量 ---
    ranks_to_test = [4, 16]  # 待测试的LoRA秩列表

    # 实验2a (容量分析) 的变量
    capacity_slices = [100, 300, 500]  # 待测试的训练数据条目数量列表

    # 实验2b (压缩分析) 的变量
    compression_levels = ['original_text', 'summ_l3_medium', 'summ_l5_extreme']  # 待测试的上下文压缩级别列表

    # 加载通用的WikiUpdate数据
    data_path = "../data/processed/wiki_update_processed.json"  # 原始处理后的数据集路径
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)  # 加载完整的JSON数据集

    # --- 外层循环：遍历所有需要测试的秩 ---
    for rank in ranks_to_test:
        print(f"\n{'=' * 25} TESTING WITH RANK (r={rank}) {'=' * 25}\n")  # 打印当前LoRA秩的测试信息

        # 动态创建当前秩的LoRA配置
        lora_config = LoraConfig(
            r=rank,  # LoRA的秩，决定适配器的低秩矩阵大小
            lora_alpha=16,  # LoRA的缩放因子，通常设为rank的2倍左右，用于缩放LoRA权重
            # target_modules: 指定哪些层需要应用LoRA适配器。这里选择了Qwen2中常见的线性层。
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,  # LoRA层的dropout比率
            bias="none",  # 不对偏置项应用LoRA
            task_type="CAUSAL_LM"  # 任务类型为因果语言模型
        )

        # --- 内层循环1：执行实验2a (容量分析) ---
        print(f"\n--- Running Experiment 2a (Capacity Analysis) for r={rank} ---")
        for num_items in capacity_slices:
            run_name = f"wiki_capacity_r{rank}_{num_items}_items"  # 当前运行的名称，用于输出目录
            print(f"\n--- Starting Training Run: {run_name} ---\n")

            # 根据当前测试的容量，截取数据集
            current_data = full_data[:num_items]
            training_samples = []  # 存储用于当前训练的格式化样本
            for item in current_data:
                context = item['original_text']  # 实验2a使用原始文本作为上下文
                # 遍历每个数据项中的“ground_truth_triplets”以提取问题和答案
                for triplet in item['ground_truth_triplets']:
                    training_samples.append({
                        "context": context,
                        "question": triplet.get('question', triplet.get('prompt', '')),  # 兼容不同键名
                        "answer": triplet.get('target', triplet.get('answer', ''))  # 兼容不同键名
                    })

            # (此部分为通用的训练流程，在两个内层循环中重复)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)  # 加载分词器
            tokenizer.pad_token = tokenizer.eos_token  # 设置填充token为EOS token
            model = AutoModelForCausalLM.from_pretrained(  # 加载预训练模型
                model_id, quantization_config=quantization_config,  # 应用量化配置
                torch_dtype=torch.bfloat16,  # 设置模型计算的数据类型
                device_map="auto",  # 自动将模型层分配到可用设备 (如GPU)
                trust_remote_code=True  # 允许加载远程代码 (Qwen2模型可能需要)
            )
            model = prepare_model_for_kbit_training(model)  # 为k-bit量化训练准备模型，冻结非LoRA参数
            model = get_peft_model(model, lora_config)  # 将LoRA适配器添加到模型
            # print(model.print_trainable_parameters()) # 可以打印可训练参数数量，检查LoRA是否生效

            dataset = Dataset.from_list(training_samples)  # 从列表创建Hugging Face Dataset
            # 使用partial创建偏函数，将tokenizer作为额外参数传递给format_data_for_qwen
            tokenized_dataset = dataset.map(partial(format_data_for_qwen, tokenizer=tokenizer))

            output_dir = f"../results/run_{run_name}"  # 设置当前训练运行的输出目录
            training_args.output_dir = output_dir  # 更新训练参数中的输出目录

            trainer = Trainer(  # 初始化Hugging Face Trainer
                model=model,  # 传入模型
                args=training_args,  # 传入训练参数
                train_dataset=tokenized_dataset,  # 传入训练数据集
                tokenizer=tokenizer,  # 传入分词器 (用于数据整理器)
                data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)  # 数据整理器，用于批量处理和填充
            )
            trainer.train()  # 开始训练
            model.save_pretrained(output_dir)  # 训练结束后保存LoRA适配器
            print(f"Finished training for {run_name}. Adapter saved.")
            # 清理内存，防止GPU OOM (Out Of Memory)
            del model, trainer, tokenizer, dataset, tokenized_dataset
            torch.cuda.empty_cache()  # 清空CUDA缓存

        # --- 内层循环2：执行实验2b (压缩分析) ---
        print(f"\n--- Running Experiment 2b (Compression Analysis) for r={rank} ---")
        for comp_type in compression_levels:
            run_name = f"wiki_compression_r{rank}_{comp_type}"  # 当前运行的名称
            print(f"\n--- Starting Training Run: {run_name} ---\n")

            training_samples = []
            for item in full_data:  # 实验2b使用全部500条数据
                context = item.get(comp_type)  # 根据压缩类型获取上下文
                if context is None or "COMPRESSION_ERROR" in str(context):  # 跳过无效或压缩失败的数据
                    continue
                for triplet in item['ground_truth_triplets']:
                    training_samples.append({
                        "context": context,
                        "question": triplet.get('question', triplet.get('prompt', '')),
                        "answer": triplet.get('target', triplet.get('answer', ''))
                    })

            if not training_samples:  # 如果没有有效的训练样本，则跳过本次训练
                print(f"No valid training samples for {run_name}. Skipping.")
                continue

            # (通用的训练流程，与实验2a相同)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=quantization_config, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            dataset = Dataset.from_list(training_samples)
            tokenized_dataset = dataset.map(partial(format_data_for_qwen, tokenizer=tokenizer))
            output_dir = f"../results/run_{run_name}"
            training_args.output_dir = output_dir
            trainer = Trainer(
                model=model, args=training_args, train_dataset=tokenized_dataset,
                tokenizer=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
            )
            trainer.train()
            model.save_pretrained(output_dir)
            print(f"Finished training for {run_name}. Adapter saved.")
            del model, trainer, tokenizer, dataset, tokenized_dataset
            torch.cuda.empty_cache()  # 清空CUDA缓存

    print("\n\nAll automated Experiment 2 training runs are complete!")  # 所有实验完成


# 确保只有在直接运行脚本时才执行main函数
if __name__ == '__main__':
    main()
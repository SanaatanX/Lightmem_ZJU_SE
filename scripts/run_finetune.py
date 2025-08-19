# 导入所有必需的库
import json  # 用于读取和解析JSON格式的数据集文件
import os    # 用于与操作系统交互，例如处理文件路径 (尽管在此脚本中未直接使用，但通常是好习惯)
from functools import partial  # 用于创建偏函数，方便地将带有固定参数的函数传递给.map()方法

import torch  # PyTorch库，是所有模型训练和张量操作的基础
from datasets import Dataset  # Hugging Face Datasets库，用于高效地处理和操作数据集
from transformers import (
    AutoModelForCausalLM,  # 自动加载因果语言模型 (例如 Qwen2)
    AutoTokenizer,         # 自动加载与模型匹配的分词器
    TrainingArguments,     # 用于配置训练过程所有超参数的类
    Trainer,               # Hugging Face提供的用于简化模型训练的高级API
    DataCollatorForSeq2Seq,# 用于将序列到序列任务的数据整理成批次 (同样适用于仅解码器的模型训练)
    BitsAndBytesConfig     # 用于配置模型的量化参数，例如4-bit加载
)
from peft import (
    LoraConfig,            # LoRA (Low-Rank Adaptation) 的配置文件，用于PEFT
    get_peft_model,        # 将基础模型转换为PEFT模型 (应用LoRA适配器)
    prepare_model_for_kbit_training # 为k-bit量化训练准备模型的辅助函数
)


# 1. 数据格式化函数
# 这个函数至关重要，它负责将我们的原始数据 (context, question, answer) 转换成Qwen2模型在训练时期望的特定对话格式。
def format_data_for_qwen(example, tokenizer, data_type='wiki'):
    """
    将单个数据条目格式化为Qwen2模型接受的指令格式。

    Args:
        example (dict): 包含 'context', 'question', 'answer' 的单个数据字典。
        tokenizer (AutoTokenizer): 用于将文本转换为token ID的分词器。
        data_type (str): 数据集类型 ('wiki' 或 'longmem')，用于选择不同的系统提示词。

    Returns:
        dict: 包含 "input_ids" 和 "labels" 的字典，可直接用于模型训练。
    """

    # 从输入字典中提取上下文、问题和答案
    context = example['context']
    question = example['question']
    answer = example['answer']

    # 根据数据集类型，定义不同的系统提示 (System Prompt)，指导模型扮演特定角色。
    if data_type == 'wiki':
        system_prompt = "You are a helpful assistant that answers questions based on the provided context."
        instruction = f"Based on the following information, answer the question.\n\nInformation:\n```{context}```\n\nQuestion:\n{question}"
    else:  # 'longmem' 数据集
        system_prompt = "You are a helpful assistant that answers questions based on a conversation history."
        instruction = f"Here is a conversation history:\n\n```{context}```\n\nBased on the conversation, answer the following question:\n{question}"

    # 构建Qwen2推荐的多轮对话消息列表。这是最新的、最标准的与指令微调模型交互的方式。
    # 格式为 [{"role": "...", "content": "..."}, ...]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": answer} # 在训练时，我们需要提供答案，模型会学习生成这个答案。
    ]

    # 使用分词器的 `apply_chat_template` 方法。这是一个强大的功能，它会自动处理特殊标记 (如 <|im_start|>, <|im_end|>)，
    # 将消息列表正确地转换为模型可以理解的token ID序列。
    tokenized_example = tokenizer.apply_chat_template(
        messages,
        tokenize=True,             # 明确指示要进行分词
        add_generation_prompt=False, # 训练时必须为False！这确保了 assistant 的回答部分也会被包含在损失计算中。如果为True，则模板会在末尾添加一个提示，让模型开始生成，这适用于推理而非训练。
        padding='max_length',      # 将所有序列填充到相同的长度 (max_length)
        truncation=True,           # 如果序列超过max_length，则进行截断
        max_length=1024            # 定义序列的最大长度，需要根据GPU显存大小进行调整
    )

    # 在因果语言模型训练中，`labels` 通常就是 `input_ids` 的一个副本。
    # 模型内部的注意力掩码会确保在预测每个token时，只能看到它之前的token。
    # 对于padding部分的token，其label会被设置为一个特殊值（通常是-100），以便在计算损失时被忽略。`DataCollator`会自动处理这个。
    return {"input_ids": tokenized_example, "labels": tokenized_example.copy()}



# 2. 主训练逻辑
def main():
    # --- 1. 全局配置：模型、量化、LoRA、训练参数 ---
    # 这些配置在所有的训练任务中保持不变。

    # 指定要微调的基础模型路径
    model_id = "../Qwen2.5-3B-Instruct"

    # 量化配置 (QLoRA的核心):
    # 使用`BitsAndBytesConfig`来实现4-bit量化加载模型，极大地减少了显存占用。
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                  # 启用4-bit量化加载
        bnb_4bit_quant_type="nf4",          # 使用NF4 (NormalFloat4) 量化类型，这是一种专为正态分布权重设计的先进类型
        bnb_4bit_compute_dtype=torch.bfloat16, # 在计算时，将权重从4-bit反量化为bfloat16类型以进行矩阵乘法，以保证精度和性能
        bnb_4bit_use_double_quant=True      # 使用双重量化，进一步节省少量显存
    )

    # LoRA配置 (PEFT的核心):
    # 定义了如何将低秩适配器（adapter）应用到模型中。我们只训练这些适配器，而不是整个模型。
    lora_config = LoraConfig(
        r=8,                               # LoRA矩阵的秩 (rank)。r越小，参数越少，但可能牺牲性能。r=8或16是常见选择。
        lora_alpha=16,                     # LoRA的缩放因子。通常设置为r的两倍。
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 指定要应用LoRA的目标模块。这里选择了注意力层和前馈网络中的所有线性层。
        lora_dropout=0.05,                 # 在LoRA层上应用的dropout，防止过拟合
        bias="none",                       # 是否训练bias。通常设置为"none"。
        task_type="CAUSAL_LM"              # 任务类型，对于Qwen2这类模型，应设为"CAUSAL_LM" (因果语言建模)
    )

    # 训练参数配置:
    # 使用`TrainingArguments`类来集中管理所有训练超参数。
    training_args = TrainingArguments(
        per_device_train_batch_size=48,    # 每个GPU的训练批量大小
        gradient_accumulation_steps=1,   # 梯度累积步数。 (batch_size * accumulation_steps) 构成了有效批量大小。
        num_train_epochs=3,                # 训练的总轮数
        learning_rate=2e-4,                # 学习率。对于LoRA微调，通常可以使用比全量微调更高的学习率。
        bf16=True,                         # 启用bfloat16混合精度训练，可以加速并减少显存，尤其适用于Ampere及更新架构的GPU。
        logging_steps=20,                  # 每隔多少步记录一次日志 (如loss)
        save_strategy="epoch",             # 模型保存策略，这里是每个epoch保存一次
        save_total_limit=1,                # 最多保留几个checkpoint，防止占满硬盘
        report_to="none",                  # 禁用向W&B等平台报告，简化本地运行
        torch_compile=True                 # (实验性功能) 尝试使用torch.compile() JIT编译器来加速模型，可能带来显著的速度提升
    )

    # --- 2. 训练流程：循环执行所有训练任务 ---
    # 脚本的核心逻辑是一个嵌套循环，外层循环处理数据集，内层循环处理该数据集下的不同压缩版本。

    # === 阶段 2.1: 处理 WikiUpdate 数据集的剩余任务 ===
    print(f"\n{'=' * 20} STAGE 2.1: Processing remaining WikiUpdate tasks {'=' * 20}")
    data_key_wiki = "wiki"
    data_path_wiki = "../data/processed/wiki_update_processed.json"
    # 定义要在此阶段处理的上下文压缩类型
    compression_types_wiki_remaining = ['summ_l5_extreme', 'ext']
    # 加载完整的JSON数据到内存
    with open(data_path_wiki, 'r', encoding='utf-8') as f:
        full_data_wiki = json.load(f)

    # 内层循环：遍历每一种压缩类型
    for comp_type in compression_types_wiki_remaining:
        run_name = f"{data_key_wiki}_{comp_type}" # 为当前训练任务创建一个唯一的名称
        print(f"\n--- Starting Training Run: {run_name} ---\n")

        # a. 准备当前任务的数据
        training_samples = []
        for item in full_data_wiki:
            context = item.get(comp_type) # 获取指定压缩类型的上下文内容
            # 数据清洗：如果上下文不存在或标记为压缩错误，则跳过该条目
            if context is None or "COMPRESSION_ERROR" in str(context): continue
            # Wiki数据集的每个item包含多个问答对 (triplets)
            for triplet in item['ground_truth_triplets']:
                training_samples.append({
                    "context": context,
                    "question": triplet.get('question', triplet.get('prompt', '')), # 兼容不同的key名
                    "answer": triplet.get('target', triplet.get('answer', ''))      # 兼容不同的key名
                })
        # 如果没有有效的样本，则跳过此次训练
        if not training_samples:
            print(f"Skipping {run_name} due to no valid training samples.");
            continue

        # b. 为当前任务重新加载模型和分词器
        #    这是关键一步，确保每次微调都从干净的、未经修改的基础模型开始。
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token # 设置pad_token为eos_token，这是处理仅解码器模型的常见做法

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config, # 应用4-bit量化配置
            torch_dtype=torch.bfloat16,              # 指定模型权重的数据类型
            device_map="auto",                       # 自动将模型分片加载到可用的GPU上
            trust_remote_code=True
        )

        # c. 应用PEFT
        model = prepare_model_for_kbit_training(model) # 准备量化后的模型以进行训练
        model = get_peft_model(model, lora_config)     # 将LoRA适配器应用到模型上，现在模型变成了PeftModel

        # d. 创建和处理数据集
        dataset = Dataset.from_list(training_samples) # 将Python列表转换为Hugging Face Dataset对象
        # 使用.map()方法批量应用格式化函数，partial确保tokenizer和data_type参数被正确传递
        tokenized_dataset = dataset.map(partial(format_data_for_qwen, tokenizer=tokenizer, data_type=data_key_wiki))

        # e. 初始化并开始训练
        output_dir = f"../results/run_{run_name}" # 为当前运行设置输出目录
        training_args.output_dir = output_dir     # 更新训练参数中的输出目录

        trainer = Trainer(
            model=model,                       # 传入PEFT模型
            args=training_args,                # 传入训练配置
            train_dataset=tokenized_dataset,   # 传入处理好的训练集
            tokenizer=tokenizer,               # 传入分词器
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8) # 数据整理器，它会负责动态填充(padding)批次中的数据，提高效率
        )
        print(f"Starting trainer for {run_name}. Results will be saved to {output_dir}")
        trainer.train() # 启动训练过程！

        # f. 保存结果并清理内存
        model.save_pretrained(output_dir) # 训练结束后，仅保存训练好的LoRA适配器权重到输出目录
        print(f"Finished training for {run_name}. Final adapter saved.")

        # 显式删除不再需要的对象并清空CUDA缓存，为下一次循环释放GPU显存
        del model, trainer, tokenizer, dataset, tokenized_dataset
        torch.cuda.empty_cache()

    # === 阶段 2.2: 处理 LongMemEval 数据集的所有任务 ===
    print(f"\n{'='*20} STAGE 2.2: Processing all LongMemEval tasks {'='*20}")
    data_key_longmem = "longmem"
    data_path_longmem = "../data/processed/longmem_eval_processed.json"
    # 定义要为LongMem数据集处理的所有压缩类型
    compression_types_longmem_all = [
        'original_text', 'summ_l1_slight', 'summ_l2_light', 'summ_l3_medium',
        'summ_l4_heavy', 'summ_l5_extreme', 'ext'
    ]
    with open(data_path_longmem, 'r', encoding='utf-8') as f:
        full_data_longmem = json.load(f)

    # 循环遍历所有LongMem的压缩类型
    for comp_type in compression_types_longmem_all:
        run_name = f"{data_key_longmem}_{comp_type}"
        print(f"\n--- Starting Training Run: {run_name} ---\n")

        # a. 准备数据 (与阶段2.1类似，但数据结构更简单)
        training_samples = []
        for item in full_data_longmem:
            context = item.get(comp_type)
            if context is None or "COMPRESSION_ERROR" in str(context): continue
            # LongMem每个item只有一个问答对
            training_samples.append({
                "context": context,
                "question": item['question'],
                "answer": str(item['answer']) # 确保答案是字符串格式
            })
        if not training_samples:
            print(f"Skipping {run_name} due to no valid training samples."); continue

        # b, c, d, e, f: 重复与阶段2.1完全相同的模型加载、PEFT应用、数据处理、训练和保存流程
        # 唯一的细微差别是在调用 .map() 时传入 `data_type=data_key_longmem`，以使用正确的系统提示词。
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        dataset = Dataset.from_list(training_samples)
        tokenized_dataset = dataset.map(partial(format_data_for_qwen, tokenizer=tokenizer, data_type=data_key_longmem))
        output_dir = f"../results/run_{run_name}"
        training_args.output_dir = output_dir
        trainer = Trainer(
            model=model, args=training_args, train_dataset=tokenized_dataset,
            tokenizer=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
        )
        print(f"Starting trainer for {run_name}. Results will be saved to {output_dir}")
        trainer.train()
        model.save_pretrained(output_dir)
        print(f"Finished training for {run_name}. Final adapter saved.")
        del model, trainer, tokenizer, dataset, tokenized_dataset
        torch.cuda.empty_cache()

    print("\n\nAll Stage 2 training runs are complete!")



if __name__ == '__main__':
    main()
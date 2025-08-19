import json  # 用于处理JSON格式的数据
import os  # 用于操作系统相关功能，例如路径操作 (在本脚本中主要用于构建输出路径)
from functools import partial  # 用于创建偏函数，允许在map函数中传递额外的固定参数

import torch  # PyTorch深度学习框架
from datasets import Dataset  # Hugging Face Datasets库中的Dataset类，用于高效处理和构建数据集
from transformers import (  # Hugging Face Transformers库，提供大模型相关工具
    AutoModelForCausalLM,  # 用于自动加载因果语言模型（如Qwen2）
    AutoTokenizer,  # 用于自动加载与模型对应的分词器
    TrainingArguments,  # 定义训练过程中的各种超参数和配置
    Trainer,  # Hugging Face提供的高级训练API，简化训练循环
    DataCollatorForSeq2Seq,  # 序列到序列任务的数据整理器，用于批处理和填充操作
    BitsAndBytesConfig  # 用于配置bitsandbytes库的量化参数，支持4位或8位量化
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # PEFT (Parameter-Efficient Fine-Tuning) 库


# LoraConfig: LoRA适配器的配置类
# get_peft_model: 将LoRA适配器集成到原始模型中
# prepare_model_for_kbit_training: 准备模型进行k-bit量化训练，通常用于QLoRA

# 1. 数据格式化函数 (此函数与之前的脚本完全相同)

def format_data_for_qwen(example, tokenizer, data_type='wiki'):
    """
    将单个数据条目格式化为Qwen2模型接受的指令格式。
    此函数模拟了Qwen2的聊天模板，将上下文、问题和答案转换为模型训练所需的输入ID序列。

    Args:
        example (dict): 包含 'context', 'question', 'answer' 字段的单个数据样本。
        tokenizer: 用于将文本转换为token ID的AutoTokenizer实例。
        data_type (str): 数据类型标识符，本脚本中未直接使用此参数改变行为，但保留以备扩展。

    Returns:
        dict: 包含 'input_ids' 和 'labels' 的字典。
              'input_ids' 是编码后的输入序列。
              'labels' 是用于计算损失的目标序列，对于因果语言模型通常与'input_ids'相同。
    """
    context = example['context']  # 从数据样本中提取上下文信息
    question = example['question']  # 从数据样本中提取问题
    answer = example['answer']  # 从数据样本中提取答案，作为模型的期望输出
    system_prompt = "You are a helpful assistant that answers questions based on the provided context."  # 定义系统角色提示
    # 构造用户指令，将上下文和问题嵌入到预定义的模板中，使用Markdown代码块格式化上下文
    instruction = f"Based on the following information, answer the question.\n\nInformation:\n```{context}```\n\nQuestion:\n{question}"

    # 按照Qwen2的官方聊天模板构建对话列表
    messages = [
        {"role": "system", "content": system_prompt},  # 系统角色消息
        {"role": "user", "content": instruction},  # 用户指令消息
        {"role": "assistant", "content": answer}  # 助手（模型）响应消息，即真实答案
    ]

    # 使用tokenizer的apply_chat_template方法将结构化对话转换为模型可接受的token ID序列
    tokenized_example = tokenizer.apply_chat_template(
        messages,
        tokenize=True,  # 执行分词操作
        add_generation_prompt=False,  # 训练时不需要在序列末尾添加生成提示
        padding='max_length',  # 对序列进行填充，使其达到最大长度
        truncation=True,  # 截断超出最大长度的序列
        max_length=1024  # 定义序列的最大长度，超出此长度的将被截断
    )
    # 对于因果语言模型（Causal LM），训练时通常将输入ID作为标签，模型学习预测序列中的下一个token
    return {"input_ids": tokenized_example, "labels": tokenized_example.copy()}


# 2. 实验B的专属主训练逻辑

def main():
    """
    此脚本专门用于执行实验B：参数存储容量分析 (Parameter Storage Capacity Analysis)。
    它旨在通过使用不同的、更精细的数据量切片来训练模型，从而深入分析LoRA模型在固定秩下
    对知识的存储和学习能力。
    """
    # --- 模型、量化、LoRA、训练参数等所有配置 ---
    # (这部分与我们最终为H800优化的版本完全相同，无需改动)
    model_id = "../Qwen2.5-3B-Instruct"  # 预训练模型的本地路径或Hugging Face模型库ID
    quantization_config = BitsAndBytesConfig(  # BitsAndBytes配置，用于4位量化 (QLoRA)
        load_in_4bit=True,  # 启用4位加载
        bnb_4bit_quant_type="nf4",  # 使用NF4量化类型，适用于非对称数据分布
        bnb_4bit_compute_dtype=torch.bfloat16,  # 量化计算时的数据类型，bfloat16有助于保持精度
        bnb_4bit_use_double_quant=True  # 启用双重量化，进一步降低内存占用
    )
    r = 4  # 固定LoRA秩为4，这是实验B的特点，只分析不同数据量下的性能
    lora_config = LoraConfig(  # LoRA适配器配置
        r=r,  # LoRA秩
        lora_alpha=16,  # LoRA缩放因子，通常设置为秩的两倍左右
        # target_modules: 指定哪些模型层会应用LoRA适配器。这里选择了Qwen2中大部分重要的线性层。
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,  # LoRA层中的dropout比率，防止过拟合
        bias="none",  # 不对偏置（bias）项应用LoRA
        task_type="CAUSAL_LM"  # 任务类型为因果语言模型
    )
    training_args = TrainingArguments(  # 训练参数配置
        per_device_train_batch_size=48,  # 每个GPU设备的训练批次大小 (已针对H800 GPU优化)
        gradient_accumulation_steps=1,  # 梯度累积步数，1表示不累积
        num_train_epochs=3,  # 总训练epoch数
        learning_rate=2e-4,  # 初始学习率
        bf16=True,  # 启用bfloat16混合精度训练，以提高训练速度和减少内存消耗
        logging_steps=20,  # 每20步记录一次训练日志
        save_strategy="epoch",  # 每个epoch结束时保存模型检查点
        save_total_limit=1,  # 最多保留一个最新的模型检查点
        report_to="none",  # 不向任何外部报告工具（如Weights & Biases、TensorBoard）报告
        torch_compile=True,  # 启用PyTorch 2.0的torch.compile进行图编译，加速训练
    )

    print(f"\n{'=' * 20} Starting Experiment B: Parameter Storage Capacity Analysis (High Granularity) {'=' * 20}")

    data_key = "wiki"  # 数据集的类型标识，本脚本中始终使用wiki数据
    data_path = "../data/processed/wiki_update_processed.json"  # 预处理后的WikiUpdate数据集路径
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)  # 加载完整的JSON格式数据集

    # 使用更精细的数据切片来训练n个模型
    data_slices_to_train = [100, 300, 500]  # 定义要测试的训练数据量切片

    # 循环遍历每个数据量切片，进行独立的训练
    for num_items in data_slices_to_train:
        run_name = f"wiki_capacity_{num_items}_items"  # 为当前训练运行生成一个描述性名称
        print(f"\n--- Starting Training Run for {num_items} items: {run_name} ---\n")

        current_training_data = full_data[:num_items]  # 根据当前num_items截取数据集
        training_samples = []  # 存储用于当前训练的格式化样本
        # 遍历当前截取的数据，提取原始上下文和所有的问答三元组
        for item in current_training_data:
            context = item['original_text']  # 实验B始终使用原始文本作为上下文
            for triplet in item['ground_truth_triplets']:
                training_samples.append({
                    "context": context,
                    "question": triplet.get('question', triplet.get('prompt', '')),  # 兼容不同键名以获取问题
                    "answer": triplet.get('target', triplet.get('answer', ''))  # 兼容不同键名以获取答案
                })

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)  # 加载分词器
        tokenizer.pad_token = tokenizer.eos_token  # 设置填充token为EOS token，Qwen2通常这样设置

        model = AutoModelForCausalLM.from_pretrained(  # 加载预训练的因果语言模型
            model_id, quantization_config=quantization_config,  # 应用4位量化配置
            torch_dtype=torch.bfloat16,  # 指定模型计算数据类型为bfloat16
            device_map="auto",  # 自动将模型层分布到可用设备 (如所有GPU)
            trust_remote_code=True,  # 允许加载模型仓库中的自定义代码
        )
        model = prepare_model_for_kbit_training(model)  # 为量化训练准备模型，通常会冻结除LoRA层外的所有参数
        model = get_peft_model(model, lora_config)  # 将LoRA适配器添加到模型中
        # print(model.print_trainable_parameters()) # 可以取消注释此行以打印可训练参数数量，验证LoRA是否正确应用

        dataset = Dataset.from_list(training_samples)  # 将Python列表转换为Hugging Face Dataset对象
        tokenized_dataset = dataset.map(  # 对数据集中的每个样本应用format_data_for_qwen函数进行分词和格式化
            partial(format_data_for_qwen, tokenizer=tokenizer, data_type=data_key)  # 使用partial函数传递tokenizer和data_type
        )

        output_dir = f"../results/run_{run_name}_{r}"  # 定义当前训练运行的输出目录，包含秩r信息
        training_args.output_dir = output_dir  # 更新TrainingArguments中的输出目录

        trainer = Trainer(  # 初始化Hugging Face Trainer
            model=model,  # 训练模型
            args=training_args,  # 训练参数
            train_dataset=tokenized_dataset,  # 训练数据集
            tokenizer=tokenizer,  # 分词器 (DataCollatorForSeq2Seq会用到)
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)  # 数据整理器，确保批次中的序列被正确填充
        )
        print(f"Starting trainer for {run_name}. Results will be saved to {output_dir}")
        trainer.train()  # 启动模型训练过程

        model.save_pretrained(output_dir)  # 训练完成后保存LoRA适配器的权重
        print(f"Finished training for {run_name}. Adapter saved to {output_dir}")
        # 显式删除模型和相关对象，并清空CUDA缓存，以释放GPU内存，为下一次训练做好准备
        del model, trainer, tokenizer, dataset, tokenized_dataset
        torch.cuda.empty_cache()

    print("\n\nAll high-granularity capacity experiment (Experiment B) training runs are complete!")  # 所有实验完成标志


# 确保只有在直接运行脚本时才执行main函数
if __name__ == '__main__':
    main()

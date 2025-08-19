

### **项目 e:\\ZJU\_SE\\ZJU\_SE 文件结构说明文档**

#### **一、 项目根目录 (Project Root)**

项目的根目录 e:\\ZJU\_SE\\ZJU\_SE\\ 包含了所有实验的核心资产和脚本。

* data/: 存放所有原始数据和经过预处理的数据。  
* scripts/: 存放所有用于执行实验的Python和Shell脚本。  
* results/: 存放所有模型微调后产出的LoRA适配器。  
* eval\_outputs/: 存放所有模型评估过程中生成的预测文件和最终分数报告。  
* LongMemEval/: 从GitHub克隆的官方评测项目，我们主要使用其src/evaluation/目录下的评测脚本。  
* Qwen2.5-3B-Instruct/: 我们从Hugging Face下载的、未经任何修改的基座模型文件。  
* Pic/: (推测) 用于存放您生成的图表和报告中使用的其他图片。  
* result.md: (推测) 您用于记录和撰写结果的Markdown文件。  
* .idea/: PyCharm等JetBrains IDE自动生成的项目配置文件，可以安全地忽略。

---

### **二、 data 目录：实验的“原材料仓库”**

这个目录是我们所有分析的起点。

* WikiUpdate.json: **原始数据**。包含了500条用于事实性知识编辑任务的原始记录。  
* longmemeval\_local/:  
  * longmemeval\_oracle.json: **原始数据**。我们手动下载的LongMemEval数据集的oracle split，包含了完整的对话历史和问答对。  
* processed/: **预处理后的数据仓库**。  
  * wiki\_update\_processed.json: 由prepare\_data.py生成的、包含了WikiUpdate原始文本及其所有6个压缩版本（5摘要+1抽取）的JSON文件。  
  * longmem\_eval\_processed.json: 由prepare\_data.py生成的、包含了LongMemEval原始对话及其所有6个压缩版本的JSON文件。

---

### **三、 scripts 目录：实验的“控制中心”**

这个目录包含了驱动整个实验流程的所有自动化脚本。

| 脚本文件名                        | 核心功能说明                                                      |
|:---------------------------- |:----------------------------------------------------------- |
| prepare\_data.py             | **数据预处理工厂**。负责加载原始数据，并行调用LLM API，生成所有压缩版本的数据。               |
| calculate\_trr.py            | **压缩比计算器**。负责计算每个压缩版本的Token缩减率(TRR)。                        |
| run\_finetune.py             | **实验A训练引擎**。专门用于在r=8的条件下，训练WikiUpdate和LongMemEval所有压缩级别的模型。 |
| run\_finetune\_exp\_b.py     | **实验B训练引擎**。专门用于在r=8的条件下，训练WikiUpdate在不同知识量（50-500条）下的模型。   |
| run\_finetune\_exp2\_auto.py | **任务2正交实验引擎**。全自动地执行r=4和r=16下的所有正交实验（容量分析与压缩分析）。            |
| evaluate\_models.py          | **模型评估器（完整版）**。负责对**所有**训练好的模型进行评估，生成预测文件，并自动调用官方脚本评分。      |
| evaluate\_baseline.py        | **基座模型评估器**。一个独立的脚本，专门用于测试**未经微调**的基座模型的性能。                 |
| analyze\_results.py          | **最终结果分析器**。负责对WikiUpdate的评估结果进行“LLM裁判”打分，并汇总所有实验的最终性能得分。   |
| plot\_final\_report.py       | **图表生成器**。负责将所有最终的性能数据，绘制成专业的、可用于报告的图表。                     |
| manual\_test.py              | **手动诊断工具**。用于加载单个模型，对其进行提问，并观察其具体回答，以进行定性分析和问题诊断。           |
| start\_training.sh           | (推测) 用于在服务器上以“守护进程”模式，一键式地启动环境配置和训练流程的Shell脚本。              |

---

### **四、 results 目录：实验的“产成品仓库”**

这个目录存放着我们整个项目**最宝贵的资产**——所有训练好的LoRA模型适配器。每一个子文件夹都代表一次独立的、成功的微调实验。

| 文件夹命名格式                                 | 示例                                   | 说明                                       |
|:--------------------------------------- |:------------------------------------ |:---------------------------------------- |
| run\_wiki\_{comp\_type}                 | run\_wiki\_summ\_l3\_medium          | **实验A**：r=8，在WikiUpdate的中度摘要上训练的模型。      |
| run\_longmem\_{comp\_type}              | run\_longmem\_summ\_l2\_light        | **实验A**：r=8，在LongMemEval的轻度摘要上训练的模型。     |
| run\_wiki\_capacity\_{num}\_items       | run\_wiki\_capacity\_250\_items      | **实验B**：r=8，在WikiUpdate的前250条数据上训练的模型。   |
| run\_wiki\_capacity\_r{R}\_{num}\_items | run\_wiki\_capacity\_r16\_500\_items | **任务2a**：r=16，在WikiUpdate的前500条数据上训练的模型。 |
| run\_wiki\_compression\_r{R}\_{comp}    | run\_wiki\_compression\_r4\_ext      | **任务2b**：r=4，在WikiUpdate的抽取式数据上训练的模型。    |

---

### **五、 eval\_outputs 目录：实验的“成绩单与答卷”**

这个目录存放着所有模型评估过程中产生的中间文件和最终结果。

| 文件名                                         | 内容说明                                                                                                |
|:------------------------------------------- |:--------------------------------------------------------------------------------------------------- |
| eval\_wiki\_all\_predictions.json           | 包含了**所有**WikiUpdate相关模型（A, B, 2a, 2b）在完整测试集上的\*\*“原始答卷”\*\*（模型生成的具体答案）。 |
| eval\_wiki\_all\_predictions\_scored.json   | 上述文件的 “已批阅版” ，为每一条回答都增加了"score"字段。                                                                  |
| final\_all\_models\_aggregated\_scores.json | **最终成绩汇总**。包含了每一个WikiUpdate相关模型的**最终平均分**。                                                          |
| preds\_{model\_name}.jsonl                  | 为LongMemEval的**每一个模型**生成的、符合官方评测脚本要求的**预测文件**（.jsonl格式）。                                            |
| preds\_{...}.jsonl.eval-results-{judge}     | 官方评测脚本运行后，为每一个预测文件生成的**详细评分日志**。                                                                    |
| eval\_longmem\_all\_scores.json             | **最终成绩汇总**。汇总了所有LongMemEval模型由官方脚本评测出的**最终分数报告**。                                                   |

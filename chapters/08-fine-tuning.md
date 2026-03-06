# 第八章：模型微调（Fine-tuning）

## 8.1 什么是微调

**微调（Fine-tuning）** 是指在预训练模型的基础上，使用特定领域的数据集进行进一步训练，使模型适应特定任务或领域的过程。

### 为什么需要微调？

1. **领域适配**：通用预训练模型在特定领域（如医疗、法律、金融）表现可能不佳
2. **任务专精**：让模型更好地完成特定任务（如情感分析、命名实体识别）
3. **风格模仿**：让模型学习特定的写作风格或对话风格
4. **知识更新**：注入预训练时不存在的新知识

### 微调 vs 预训练

| 对比项 | 预训练（Pre-training） | 微调（Fine-tuning） |
|--------|----------------------|-------------------|
| 数据量 | 海量（TB 级） | 较小（MB-GB 级） |
| 计算成本 | 极高（数千 GPU 小时） | 较低（几到几十 GPU 小时） |
| 目标 | 学习通用语言表示 | 适应特定任务/领域 |
| 数据标注 | 无监督/自监督 | 通常需要标注数据 |

---

## 8.2 微调的主要方法

### 8.2.1 全量微调（Full Fine-tuning）

更新模型的所有参数。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# 准备数据集
train_dataset = ...  # 你的训练数据

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
)

# 创建 Trainer 并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**优点**：
- 性能通常最好
- 简单直接

**缺点**：
- 需要大量显存
- 容易过拟合小数据集
- 训练时间长

---

### 8.2.2 参数高效微调（PEFT）

只更新模型的一小部分参数，大幅降低计算成本。

#### LoRA（Low-Rank Adaptation）

**核心思想**：在模型的权重矩阵上添加低秩适配器，只训练适配器参数。

```
原始权重：W ∈ R^(d×k)
LoRA 更新：W' = W + ΔW = W + BA
其中：B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
```

```python
from peft import LoraConfig, get_peft_model, TaskType

# 配置 LoRA
lora_config = LoraConfig(
    r=8,  # 秩，通常 8-64
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 要适配的模块
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数比例

# 正常训练
trainer.train()
```

**常见参数配置**：
- `r`: 8, 16, 32（越大表达能力越强，但参数越多）
- `lora_alpha`: 通常是 r 的 2-4 倍
- `target_modules`: Qwen 系列常用 `["q_proj", "v_proj", "k_proj", "o_proj"]`

**优点**：
- 显存需求降低 60-80%
- 训练速度更快
- 不易过拟合
- 可保存多个适配器切换使用

**缺点**：
- 性能略低于全量微调（但差距很小）

---

#### QLoRA（Quantized LoRA）

LoRA + 量化，进一步降低显存需求。

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto"
)

# 准备 k-bit 训练
model = prepare_model_for_kbit_training(model)

# 添加 LoRA 并训练
model = get_peft_model(model, lora_config)
trainer.train()
```

**显存对比**（以 7B 模型为例）：
- 全量微调：~80GB
- LoRA：~20GB
- QLoRA：~8GB

---

### 8.2.3 其他 PEFT 方法

| 方法 | 核心思想 | 适用场景 |
|------|---------|---------|
| **Adapter** | 插入小型神经网络模块 | 多任务学习 |
| **Prefix Tuning** | 学习可训练的前缀向量 | 生成任务 |
| **Prompt Tuning** | 学习软提示（soft prompts） | 分类/生成 |
| **IA³** | 学习激活缩放因子 | 超低参数场景 |

---

## 8.3 数据准备

### 8.3.1 数据格式

#### 指令微调格式（Instruction Tuning）

```json
[
  {
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
  },
  {
    "instruction": "将以下文本翻译成英文",
    "input": "今天天气很好",
    "output": "The weather is very nice today."
  }
]
```

#### 对话格式

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手"},
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"}
    ]
  }
]
```

#### 续写格式

```json
[
  {
    "text": "机器学习是人工智能的核心技术之一。它通过..."
  }
]
```

---

### 8.3.2 数据质量要求

1. **多样性**：覆盖不同场景、风格、难度
2. **准确性**：答案正确，无事实错误
3. **一致性**：格式统一，风格一致
4. **适量性**：
   - 简单任务：100-1000 条
   - 复杂任务：1000-10000 条
   - 领域适配：5000-50000 条

---

### 8.3.3 数据预处理

```python
def preprocess(example):
    # 拼接指令和输入
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example['input']:
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Output:\n{example['output']}"
    
    # 分词
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    # 设置 label（用于计算 loss）
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

train_dataset = train_dataset.map(preprocess, remove_columns=original_columns)
```

---

## 8.4 训练技巧

### 8.4.1 学习率设置

| 微调类型 | 推荐学习率 |
|---------|-----------|
| 全量微调 | 1e-5 ~ 5e-5 |
| LoRA | 1e-4 ~ 5e-4 |
| QLoRA | 2e-4 ~ 1e-3 |

**学习率调度**：
```python
training_args = TrainingArguments(
    ...
    learning_rate=2e-4,
    lr_scheduler_type="cosine",  # 余弦退火
    warmup_ratio=0.03,  # 3% 步数用于预热
)
```

---

### 8.4.2 防止过拟合

1. **早停（Early Stopping）**
```python
training_args = TrainingArguments(
    ...
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    save_total_limit=2,
)
```

2. **权重衰减**
```python
training_args = TrainingArguments(
    ...
    weight_decay=0.01,
)
```

3. **Dropout**
```python
lora_config = LoraConfig(
    ...
    lora_dropout=0.1,
)
```

4. **数据增强**
   - 回译（Back Translation）
   - 同义词替换
   - 随机删除/插入

---

### 8.4.3 梯度累积

显存不足时使用：

```python
training_args = TrainingArguments(
    ...
    per_device_train_batch_size=2,  # 每设备 batch
    gradient_accumulation_steps=8,   # 累积 8 步
    # 有效 batch = 2 * 8 = 16
)
```

---

## 8.5 评估方法

### 8.5.1 自动评估指标

| 指标 | 适用场景 | 说明 |
|------|---------|------|
| **Perplexity** | 语言建模 | 越低越好 |
| **ROUGE** | 摘要生成 | 衡量 n-gram 重叠 |
| **BLEU** | 翻译 | 衡量精度 |
| **Accuracy** | 分类任务 | 正确率 |
| **F1 Score** | 分类/抽取 | 精确率 + 召回率 |

```python
from evaluate import load

rouge = load("rouge")
predictions = ["模型输出 1", "模型输出 2"]
references = ["标准答案 1", "标准答案 2"]

results = rouge.compute(predictions=predictions, references=references)
print(results)
```

---

### 8.5.2 人工评估

设计评估维度：
1. **准确性**：答案是否正确
2. **相关性**：是否回答了问题
3. **流畅性**：语言是否自然
4. **完整性**：信息是否充分
5. **安全性**：是否有害内容

---

### 8.5.3 基准测试

使用公开评测集：
- **C-Eval**：中文综合评测
- **CMMLU**：中文多任务语言理解
- **HumanEval**：代码生成
- **GSM8K**：数学推理

---

## 8.6 常见问题与解决方案

### Q1: 训练 loss 不下降

**可能原因**：
- 学习率太小
- 数据有问题（格式错误、标签错误）
- 模型容量不足

**解决方案**：
- 调大学习率
- 检查数据预处理
- 换更大的基座模型

---

### Q2: 训练 loss 下降但验证 loss 上升

**原因**：过拟合

**解决方案**：
- 减少训练轮数
- 增加正则化（dropout、weight decay）
- 增加训练数据
- 使用早停

---

### Q3: 显存不足（OOM）

**解决方案**：
1. 减小 batch size + 梯度累积
2. 使用 LoRA/QLoRA
3. 使用梯度检查点（gradient checkpointing）
4. 混合精度训练（FP16/BF16）

```python
training_args = TrainingArguments(
    ...
    fp16=True,  # 或 bf16=True
    gradient_checkpointing=True,
)
```

---

### Q4: 生成内容重复/退化

**原因**：
- 训练数据质量差
- 学习率过大
- 训练轮数过多

**解决方案**：
- 清洗训练数据
- 降低学习率
- 减少训练轮数
- 调整生成参数（temperature、top_p）

---

## 8.7 实战案例

### 案例 1：客服机器人微调

**目标**：让模型学习公司的产品知识和客服话术

**数据**：5000 条历史客服对话

**配置**：
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

training_args = TrainingArguments(
    output_dir="./customer-service-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
)
```

---

### 案例 2：医疗问答微调

**目标**：让模型具备医疗领域知识

**挑战**：
- 数据敏感性高
- 准确性要求极高
- 需要避免给出错误医疗建议

**方案**：
1. 使用专业医疗数据集（如 PubMedQA）
2. 添加系统提示："我是 AI 助手，不能替代专业医疗建议"
3. 对不确定问题引导用户咨询医生
4. 严格的人工审核

---

### 案例 3：代码生成微调

**目标**：提升特定编程语言的代码生成能力

**数据格式**：
```json
{
  "instruction": "用 Python 实现快速排序",
  "output": "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    ..."
}
```

**评估**：使用 HumanEval 或自定义测试用例

---

## 8.8 最佳实践总结

### ✅ 推荐做法

1. **从预训练模型开始**：不要从头训练
2. **先小后大**：先用小数据集测试流程
3. **保存检查点**：定期保存模型
4. **监控训练**：使用 TensorBoard/WandB
5. **版本管理**：记录每次实验的配置
6. **A/B 测试**：对比不同配置的效果

### ❌ 避免的坑

1. **数据泄露**：训练集和测试集重叠
2. **过拟合**：在小数据集上训练太久
3. **忽略评估**：只看 training loss
4. **盲目调参**：没有系统的实验设计
5. **忽视安全**：未过滤有害内容

---

## 8.9 工具推荐

| 工具 | 用途 | 链接 |
|------|------|------|
| **Hugging Face Transformers** | 模型加载/训练 | transformers.huggingface.co |
| **PEFT** | 参数高效微调 | github.com/huggingface/peft |
| **TRL** | RLHF 训练 | github.com/huggingface/trl |
| **Axolotl** | 微调框架 | github.com/OpenAccess-AI-Collective/axolotl |
| **LLaMA-Factory** | 一站式微调平台 | github.com/hiyouga/LLaMA-Factory |
| **WandB** | 实验追踪 | wandb.ai |

---

## 8.10 面试真题

### 真题 1：LoRA 的原理是什么？为什么有效？

**参考答案**：

LoRA 的核心思想是：模型在适应特定任务时，权重变化的内在秩（intrinsic rank）很低。

具体做法：
1. 冻结预训练权重 W
2. 添加低秩分解 ΔW = BA，其中 B∈R^(d×r), A∈R^(r×k)
3. 只训练 A 和 B，r 远小于 d 和 k

为什么有效：
1. **参数效率**：7B 模型全量微调需要 28GB 参数，LoRA(r=8) 只需几 MB
2. **无推理延迟**：训练后可合并权重：W' = W + BA
3. **模块化**：同一基座可加载多个适配器切换任务
4. **性能接近**：实验表明 LoRA 性能与全量微调相当

---

### 真题 2：如何选择合适的微调方法？

**参考答案**：

根据资源、数据、目标选择：

| 场景 | 推荐方法 |
|------|---------|
| 显存充足 + 数据量大 | 全量微调 |
| 显存有限 + 单任务 | LoRA |
| 显存非常有限 | QLoRA |
| 多任务切换 | LoRA（多个适配器） |
| 数据极少（<100 条） | Prompt Tuning / 少样本学习 |
| 需要快速实验 | LoRA/QLoRA |

实际建议：优先尝试 LoRA，成本低、效果好、易部署。

---

### 真题 3：微调数据应该准备多少？

**参考答案**：

取决于任务复杂度：

- **简单任务**（分类、情感分析）：100-1000 条
- **中等任务**（问答、摘要）：1000-10000 条
- **复杂任务**（对话、创作）：5000-50000 条
- **领域适配**：越多越好，至少 10000 条

关键原则：
1. **质量 > 数量**：1000 条高质量数据 > 10000 条低质量
2. **多样性**：覆盖不同场景和边缘情况
3. **平衡性**：各类别数据分布均衡
4. **增量迭代**：先小后大，逐步增加

---

## 本章小结

1. **微调**是让预训练模型适应特定任务的关键技术
2. **PEFT 方法**（LoRA、QLoRA）大幅降低了微调门槛
3. **数据质量**比数量更重要
4. **实验追踪**和**系统评估**不可或缺
5. 根据**资源约束**选择合适的微调策略

---

## 下一章预告

第九章将介绍 **AI Agent（智能体）**，包括：
- Agent 的核心组件
- 工具调用（Function Calling）
- 记忆与规划
- 多 Agent 协作
- Agent 开发框架

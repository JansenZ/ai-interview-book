# 第一章：AI 术语速成

> 💡 **本章目标**：看完这章，你能听懂 AI 工程师在说什么，能看懂 AI 相关的技术文档
> 
> ⏱️ **预计时间**：2-3 小时
> 
> 📌 **面试重要度**：⭐⭐⭐⭐⭐（必考）

---

## 1.1 AI/ML/DL/LLM 到底是什么关系

这是面试必问的第一个问题，90% 的候选人会搞混。

### 一张图看懂

```
人工智能 (AI - Artificial Intelligence)
│
├── 规则系统（不是学习）
│   └── 人工写 if-else 规则
│
└── 机器学习 (ML - Machine Learning) ⭐
    │
    ├── 传统机器学习
    │   ├── 线性回归（预测房价）
    │   ├── 逻辑回归（判断垃圾邮件）
    │   ├── 决策树（分类）
    │   ├── 随机森林（多个决策树）
    │   ├── SVM（支持向量机）
    │   └── K-Means（聚类）
    │
    ├── 深度学习 (DL - Deep Learning) ⭐⭐
    │   ├── CNN（卷积神经网络）- 处理图片
    │   ├── RNN（循环神经网络）- 处理序列
    │   ├── LSTM（长短期记忆）- RNN 的改进
    │   └── Transformer - 处理文本（最重要！）
    │
    └── 强化学习
        └── AlphaGo、游戏 AI

大语言模型 (LLM - Large Language Model) ⭐⭐⭐
└── 基于 Transformer 的超大规模模型
    ├── GPT 系列（OpenAI）
    ├── Claude 系列（Anthropic）
    ├── 通义千问（阿里）
    ├── 文心一言（百度）
    └── Llama 系列（Meta，开源）
```

### 用人类学习类比

```
AI（人工智能）= 让机器像人一样智能

ML（机器学习）= 让机器从例子中学习，而不是背规则
  类比：教小孩认猫
  - 传统方法：告诉小孩"猫有尖耳朵、胡须、尾巴"（规则）
  - 机器学习：给小孩看 100 张猫的照片，他自己总结规律

DL（深度学习）= 用多层神经网络学习
  类比：小孩学习更复杂的概念
  - 单层：识别猫 vs 狗
  - 多层：识别猫的品種、年龄、情绪

LLM（大语言模型）= 用 Transformer 架构 + 海量数据训练的模型
  类比：一个读了互联网上所有书的天才
  - 什么都会一点
  - 能聊天、能写代码、能做题
```

### 面试回答模板

> **面试官**：请解释一下 AI、ML、DL、LLM 的区别和联系。

**参考回答**：

```
从属关系是：AI > ML > DL > LLM

AI（人工智能）是总称，指让机器模拟人类智能的技术。

ML（机器学习）是 AI 的一个子集，核心思想是从数据中学习规律，
而不是人工写规则。比如垃圾邮件检测，传统方法要人工写规则
（包含"中奖"就是垃圾邮件），机器学习是给模型大量标注数据，
让它自己学习哪些特征代表垃圾邮件。

DL（深度学习）是机器学习的一种，用多层神经网络。
"深度"指的是层数多。深度学习的优势是能自动学习特征，
不需要人工设计。比如图片识别，传统 ML 需要人工设计特征
（边缘、纹理等），深度学习直接从原始像素学习。

LLM（大语言模型）是深度学习的一种，基于 Transformer 架构，
参数规模超大（几百亿到万亿），训练数据是互联网上的全部文本。
因为规模大、数据多，所以有很强的通用能力，能聊天、写代码、
做题等。代表产品有 GPT-4、Claude、通义千问等。

关键区别：
1. 数据量：ML 小数据可用，DL 需要大数据，LLM 需要海量数据
2. 特征工程：ML 需要人工设计，DL 自动学习
3. 通用性：ML/DL 针对特定任务，LLM 通用性强
```

---

## 1.2 模型、训练、推理、参数、权重

这些是最基础的术语，必须搞懂。

### 模型（Model）

**定义**：训练好的 AI 程序，可以用来做预测或生成。

**前端类比**：
```javascript
// 模型就像一个封装好的 npm 包
import sentimentModel from 'sentiment-analysis';

// 调用模型做预测
const result = sentimentModel.predict('这个产品太好用了！');
// 输出：{ sentiment: 'positive', score: 0.95 }
```

**常见模型类型**：
- 分类模型：判断邮件是垃圾邮件还是正常邮件
- 回归模型：预测房价
- 生成模型：写文章、生成图片
- 对话模型：聊天机器人

### 训练（Training）

**定义**：用数据"教"模型学习的过程。

**类比**：
```
训练模型 = 教小孩学习

1. 准备教材（训练数据）
   - 1000 张猫的图片 + 标签（这是猫）
   - 1000 张狗的图片 + 标签（这是狗）

2. 学习过程（训练）
   - 小孩看图片，尝试识别
   - 识别错了，纠正
   - 重复多次，逐渐提高准确率

3. 毕业（训练完成）
   - 小孩学会了认猫和狗
   - 可以识别没见过的猫狗图片
```

**训练过程详解**：
```
输入训练数据
    ↓
模型做预测
    ↓
计算误差（预测 vs 真实）
    ↓
调整参数（减少误差）
    ↓
重复以上步骤（1000-100000 次）
    ↓
模型训练完成
```

**代码示例**：
```javascript
// 伪代码，展示训练流程
const model = new NeuralNetwork();

// 训练数据
const trainingData = [
  { input: [图片 1], label: '猫' },
  { input: [图片 2], label: '狗' },
  // ... 10000 张
];

// 训练循环
for (let epoch = 0; epoch < 1000; epoch++) {
  for (const data of trainingData) {
    // 1. 模型预测
    const prediction = model.predict(data.input);
    
    // 2. 计算误差
    const error = calculateError(prediction, data.label);
    
    // 3. 调整参数（反向传播）
    model.adjustParameters(error);
  }
  
  console.log(`第${epoch}轮，误差：${error}`);
}
```

### 推理（Inference）

**定义**：用训练好的模型做预测或生成。

**类比**：
```
训练 = 学习过程（在学校读书）
推理 = 应用知识（工作干活）
```

**代码示例**：
```javascript
// 训练好的模型（推理阶段）
const trainedModel = loadModel('cat-dog-classifier');

// 推理：识别新图片
const result = trainedModel.predict(newImage);
// 输出：{ label: '猫', confidence: 0.92 }

// 对话模型推理
const response = await chatModel.generate('你好，请介绍一下自己');
// 输出：'你好！我是一个 AI 助手...'
```

**训练 vs 推理**：

| 对比项 | 训练 | 推理 |
|--------|------|------|
| 目的 | 学习规律 | 应用知识 |
| 数据 | 大量标注数据 | 单个或少量输入 |
| 计算量 | 大（需要 GPU） | 小（CPU 也可） |
| 时间 | 几小时到几个月 | 几毫秒到几秒 |
| 频率 | 一次性或偶尔 | 持续进行 |
| 前端相关 | 不直接参与 | 经常调用 |

### 参数（Parameters）

**定义**：模型内部的变量，决定模型的行为。

**类比**：
```javascript
// 模型就像一个函数
function predict(input) {
  // 参数就是函数内部的变量
  const weight1 = 0.5;  // 参数
  const weight2 = 0.3;  // 参数
  const bias = 0.1;     // 参数
  
  return input * weight1 + weight2 + bias;
}

// 训练就是找到最好的参数值
```

**参数量级**：
- 小模型：几百万参数（MB 级别）
- 中等模型：几亿参数（GB 级别）
- 大模型：几百亿到万亿参数（百 GB 到 TB 级别）

**GPT 系列参数量**：
```
GPT-1:    1.17 亿参数
GPT-2:    15 亿参数
GPT-3:    1750 亿参数
GPT-4:    未公开（估计万亿级别）
```

### 权重（Weights）

**定义**：参数的具体数值。

**详细解释**：
```
神经网络由多层神经元组成：

输入层 → 隐藏层 1 → 隐藏层 2 → 输出层

每两层之间有连接，每个连接有一个权重值。

权重决定：
- 输入信号的重要性
- 信号如何传递到下一层

训练过程就是不断调整权重，让模型输出更准确。
```

**权重示例**：
```javascript
// 简化版神经网络
class SimpleNeuralNetwork {
  constructor() {
    // 初始化权重（随机值）
    this.weights = [
      [0.1, 0.2, 0.3],  // 第一层权重
      [0.4, 0.5],       // 第二层权重
    ];
    this.biases = [0.1, 0.2];
  }
  
  predict(input) {
    // 前向传播：输入 × 权重 + 偏置
    let hidden = input.map((x, i) => x * this.weights[0][i]);
    let output = hidden.reduce((a, b) => a + b) + this.biases[0];
    return output;
  }
  
  // 训练时调整权重
  adjustWeights(error) {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = this.weights[i].map(w => w - error * 0.01);
    }
  }
}
```

---

## 1.3 Token、Context Window、Temperature

这三个是调用大模型 API 时最常用的概念。

### Token

**定义**：大模型处理文本的最小单位。

**重要**：大模型不是按"字"或"词"计算，而是按 Token 计算。

**Token 换算**：
```
英文：
- 1 个单词 ≈ 1.3 个 Token
- "Hello, world!" = 4 个 Token（Hello, ,, world, !）

中文：
- 1 个汉字 ≈ 1.5 个 Token
- "你好，世界！" = 6-8 个 Token

代码：
- 1 个关键字 ≈ 1 个 Token
- "function test() {}" ≈ 6 个 Token
```

**Token 计算器**：
```javascript
// 估算 Token 数量（粗略）
function estimateTokens(text, language = 'zh') {
  if (language === 'en') {
    return Math.ceil(text.split(/\s+/).length * 1.3);
  } else {
    // 中文：按字符数 × 1.5
    return Math.ceil(text.length * 1.5);
  }
}

// 示例
console.log(estimateTokens('Hello world')); // ≈ 3 tokens
console.log(estimateTokens('你好世界')); // ≈ 6 tokens
```

**为什么重要**：
1. **计费**：API 按 Token 收费
2. **限制**：模型有最大 Token 限制
3. **性能**：Token 越多，处理越慢

**计费示例**：
```
GPT-4o 价格：
- 输入：$5 / 1M tokens
- 输出：$15 / 1M tokens

1000 个中文字 ≈ 1500 tokens
成本：1500 / 1_000_000 × $5 = $0.0075（约 5 分钱）

一次对话（输入 1000 字 + 输出 500 字）：
成本 ≈ $0.01（约 7 分钱）
```

### Context Window（上下文窗口）

**定义**：模型一次能处理的最大 Token 数量。

**包含**：输入（Prompt）+ 输出（Completion）

**常见模型的 Context Window**：
```
GPT-4o:          128,000 tokens（约 10 万字）
Claude 3.5:      200,000 tokens（约 15 万字）
通义千问 Plus:    32,000 tokens（约 2 万字）
DeepSeek V2:     128,000 tokens（约 10 万字）
Llama 3:         8,000 tokens（约 6000 字）
```

**实际意义**：
```
Context Window = 模型一次能"记住"的内容量

示例：
- GPT-4o (128K)：可以处理一整本书
- Claude 3.5 (200K)：可以处理多份文档
- Llama 3 (8K)：只能处理短文章
```

**多轮对话中的 Context**：
```javascript
// 对话历史会占用 Context Window
const messages = [
  { role: 'user', content: '你好' },           // 10 tokens
  { role: 'assistant', content: '你好！...' },  // 50 tokens
  { role: 'user', content: '请问...' },         // 100 tokens
  { role: 'assistant', content: '...' },        // 200 tokens
  // ... 更多对话
  
  // 新问题时，需要带上之前的对话历史
  // 如果超过 Context Window，需要截断或总结
];

// 检查是否超出限制
const totalTokens = messages.reduce((sum, m) => sum + estimateTokens(m.content), 0);
if (totalTokens > 128000) {
  // 需要处理：删除最早的对话或总结
}
```

### Temperature（温度）

**定义**：控制输出的随机性（创造性）。

**取值范围**：0 到 2（常用 0 到 1）

**效果**：
```
Temperature = 0：
- 输出确定性最高
- 每次回答几乎一样
- 适合事实性问题、代码生成

Temperature = 0.7（默认）：
- 平衡创造性和准确性
- 适合一般对话

Temperature = 1.0+：
- 输出非常随机
- 适合创意写作、头脑风暴
```

**示例对比**：
```javascript
// 问题："天空是什么颜色的？"

// Temperature = 0
输出："天空是蓝色的。"（确定、简洁）

// Temperature = 0.7
输出："天空通常是蓝色的，但在日出和日落时会呈现橙色或红色。"

// Temperature = 1.5
输出："天空？嗯... 有时候是蓝色，有时候是灰色，有时候像被打翻的调色盘，
      有橙色、粉色、紫色... 你最喜欢什么颜色的天空？"（发散、创意）
```

**使用建议**：
```javascript
const configs = {
  // 客服场景：准确第一
  customerService: { temperature: 0.3 },
  
  // 代码生成：确定性高
  codeGeneration: { temperature: 0.2 },
  
  // 创意写作：需要灵感
  creativeWriting: { temperature: 0.8 },
  
  // 头脑风暴：越发散越好
  brainstorming: { temperature: 1.2 },
};
```

---

## 1.4 Embedding、向量、相似度计算

这是 RAG（检索增强生成）的核心概念，必须搞懂。

### Embedding（嵌入）

**定义**：把文本（或图片）转换成一串数字（向量）。

**核心思想**：语义相似的内容，Embedding 也相似。

**可视化理解**：
```
文本 → Embedding 模型 → 向量（一串数字）

"猫" → [0.1, -0.5, 0.3, 0.8, -0.2, ...]  // 768 个数字
"狗" → [0.2, -0.4, 0.2, 0.7, -0.3, ...]  // 768 个数字
"汽车" → [-0.8, 0.6, -0.5, 0.1, 0.9, ...] // 768 个数字

"猫"和"狗"都是动物，向量相似（距离近）
"猫"和"汽车"不相关，向量不相似（距离远）
```

**维度**：
```
text-embedding-3-small:  1536 维
text-embedding-3-large:  3072 维
m3e-base:                768 维
```

**代码示例**：
```javascript
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function getEmbedding(text) {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  
  return response.data[0].embedding;
}

// 使用
const catEmbedding = await getEmbedding('猫');
const dogEmbedding = await getEmbedding('狗');
const carEmbedding = await getEmbedding('汽车');

console.log(catEmbedding.length); // 1536
```

**应用场景**：
1. **语义搜索**：搜"宠物"能找到"猫"、"狗"相关文章
2. **推荐系统**：推荐相似内容
3. **RAG**：检索相关文档
4. **聚类分析**：自动分组相似内容

### 向量（Vector）

**定义**：一组数字，可以理解为多维空间中的一个点。

**直观理解**：
```
2 维向量：[x, y] → 平面上的一个点
3 维向量：[x, y, z] → 空间中的一个点
1536 维向量：→ 1536 维空间中的一个点（无法可视化，但数学上一样）
```

**向量的意义**：
```
每个维度代表一个特征：

"猫"的 Embedding（简化版）：
[
  0.1,   // 维度 1：是否有毛
  -0.5,  // 维度 2：体型大小
  0.3,   // 维度 3：是否家养
  0.8,   // 维度 4：是否喵喵叫
  -0.2,  // 维度 5：...
  ...    // 1536 个维度
]

"狗"的 Embedding：
[
  0.2,   // 维度 1：是否有毛（相似）
  -0.4,  // 维度 2：体型大小（相似）
  0.2,   // 维度 3：是否家养（相似）
  0.7,   // 维度 4：是否汪汪叫（不同）
  ...
]
```

### 相似度计算

**为什么要计算相似度**：
```
场景：用户搜索"宠物"

1. 把"宠物"转换成向量
2. 计算"宠物"向量和所有文档向量的相似度
3. 返回相似度最高的文档
```

**余弦相似度（最常用）**：

**公式**：
```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)

A · B = 点积 = Σ(Aᵢ × Bᵢ)
|A| = 向量 A 的模 = √(Σ Aᵢ²)
```

**结果范围**：
```
1：完全相同（夹角 0 度）
0：无关（夹角 90 度）
-1：完全相反（夹角 180 度）
```

**代码实现**：
```javascript
function cosineSimilarity(a, b) {
  // 1. 计算点积
  const dotProduct = a.reduce((sum, _, i) => sum + a[i] * b[i], 0);
  
  // 2. 计算模
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  
  // 3. 计算余弦相似度
  return dotProduct / (normA * normB);
}

// 测试
const vector1 = [0.1, 0.2, 0.3];
const vector2 = [0.1, 0.2, 0.3];
const vector3 = [0.9, 0.8, 0.7];

console.log(cosineSimilarity(vector1, vector2)); // 1.0（完全相同）
console.log(cosineSimilarity(vector1, vector3)); // 0.96（很相似）
```

**完整示例：语义搜索**：
```javascript
// 1. 准备文档库
const documents = [
  { id: 1, content: '猫是很可爱的宠物' },
  { id: 2, content: '狗是人类最好的朋友' },
  { id: 3, content: '汽车有四个轮子' },
  { id: 4, content: '养猫需要注意饮食' },
];

// 2. 预先计算所有文档的 Embedding
const documentEmbeddings = [];
for (const doc of documents) {
  const embedding = await getEmbedding(doc.content);
  documentEmbeddings.push({ ...doc, embedding });
}

// 3. 用户搜索
async function search(query) {
  // 3.1 问题向量化
  const queryEmbedding = await getEmbedding(query);
  
  // 3.2 计算相似度
  const results = documentEmbeddings.map(doc => ({
    ...doc,
    similarity: cosineSimilarity(queryEmbedding, doc.embedding),
  }));
  
  // 3.3 按相似度排序
  results.sort((a, b) => b.similarity - a.similarity);
  
  // 3.4 返回最相关的
  return results.slice(0, 3);
}

// 测试
const results = await search('宠物');
// 返回：文档 1（猫）、文档 2（狗）、文档 4（养猫）
// 文档 3（汽车）不相关，不会返回
```

---

## 1.5 常见模型名称和特点

面试时可能会问到具体模型，这里总结主流模型。

### 国际厂商

#### GPT 系列（OpenAI）

```
GPT-3.5 Turbo:
- 参数：约 1750 亿
- 特点：快、便宜、够用
- 适用：一般对话、简单任务
- 价格：$0.5 / 1M tokens（输入）

GPT-4 / GPT-4o:
- 参数：未公开（估计万亿级别）
- 特点：最强通用能力
- 适用：复杂任务、代码、推理
- 价格：$5 / 1M tokens（输入）

GPT-4 Vision:
- 特点：能看懂图片
- 适用：图片分析、OCR
```

#### Claude 系列（Anthropic）

```
Claude 3.5 Sonnet:
- Context: 200K tokens
- 特点：长文本处理强、代码能力强
- 适用：文档分析、代码生成
- 价格：$3 / 1M tokens（输入）

Claude 3 Opus:
- 特点：最强推理能力
- 适用：复杂分析、研究
- 价格：$15 / 1M tokens（输入）
```

#### Gemini 系列（Google）

```
Gemini Pro:
- 特点：多模态好、Google 生态
- 适用：图片理解、视频分析
- 价格：$0.5 / 1M tokens

Gemini Ultra:
- 特点：最强版本
- 适用：复杂任务
```

#### Llama 系列（Meta，开源）

```
Llama 3 8B:
- 参数：80 亿
- 特点：小、快、可本地运行
- 适用：资源受限场景

Llama 3 70B:
- 参数：700 亿
- 特点：开源最强
- 适用：自部署、定制
```

### 国内厂商

#### 通义千问（阿里）

```
Qwen-Turbo:
- 特点：快、便宜
- 价格：¥0.002 / 1K tokens

Qwen-Plus:
- 特点：平衡性能和价格
- 价格：¥0.004 / 1K tokens

Qwen-Max:
- 特点：最强版本
- 价格：¥0.02 / 1K tokens
```

#### 文心一言（百度）

```
文心一言 4.0:
- 特点：中文优化
- 价格：¥0.012 / 1K tokens
```

#### DeepSeek（深度求索）

```
DeepSeek V2:
- 特点：性价比之王
- 价格：¥0.001 / 1K tokens（输入）
- 推荐：预算有限首选
```

### 选择建议

```
追求效果：GPT-4o 或 Claude 3.5
预算有限：DeepSeek 或通义千问
国内部署：通义千问或 DeepSeek
需要开源：Llama 3
长文本：Claude 3.5（200K Context）
多模态：GPT-4V 或 Gemini
```

---

## 1.6 面试考点自测

### 基础题（必须会）

**1. 解释 AI、ML、DL、LLM 的关系**

<details>
<summary>点击查看答案</summary>

从属关系：AI > ML > DL > LLM

AI 是总称，ML 是从数据学习，DL 是用神经网络，LLM 是基于 Transformer 的大模型。

关键区别：
- 数据量：ML 小数据可用，LLM 需要海量数据
- 特征工程：ML 需要人工设计，DL 自动学习
- 通用性：ML 针对特定任务，LLM 通用
</details>

**2. 什么是 Token？1000 个中文字大约多少 Token？**

<details>
<summary>点击查看答案</summary>

Token 是大模型处理文本的最小单位。

1000 个中文字 ≈ 1500 个 Token（1 汉字 ≈ 1.5 Token）
</details>

**3. Temperature 参数有什么用？**

<details>
<summary>点击查看答案</summary>

控制输出的随机性（0-2）。

0：确定性最高，适合事实性问题
0.7：默认，平衡创造性和准确性
1.0+：随机性强，适合创意写作
</details>

**4. 什么是 Embedding？有什么用？**

<details>
<summary>点击查看答案</summary>

把文本转换成一串数字（向量）。

语义相似的内容，Embedding 也相似。

用途：语义搜索、RAG 检索、推荐系统、聚类分析。
</details>

**5. 余弦相似度的范围是多少？表示什么？**

<details>
<summary>点击查看答案</summary>

范围：-1 到 1

1：完全相同
0：无关
-1：完全相反

AI 中常用于计算文本相似度。
</details>

### 进阶题（了解加分）

**6. Context Window 是什么？为什么重要？**

<details>
<summary>点击查看答案</summary>

模型一次能处理的最大 Token 数量。

重要原因：
1. 限制了单次输入的长度
2. 多轮对话会占用 Context
3. 超过限制需要截断或总结

GPT-4o：128K，Claude 3.5：200K
</details>

**7. 训练和推理的区别？**

<details>
<summary>点击查看答案</summary>

训练：用数据学习，计算量大，一次性
推理：用模型预测，计算量小，持续进行

前端主要参与推理阶段（调用 API）
</details>

**8. 参数和权重的关系？**

<details>
<summary>点击查看答案</summary>

参数是总称，权重是参数的一种。

神经网络的参数包括：
- 权重（Weights）：连接的强度
- 偏置（Biases）：激活阈值
</details>

---

## 本章小结

### 核心概念

```
AI/ML/DL/LLM：从属关系，一定要搞懂
Token：计费单位，1 中文 ≈ 1.5 Token
Context Window：一次能处理的最大长度
Temperature：控制输出随机性
Embedding：文本→向量，用于相似度计算
```

### 面试必背

```
1. AI > ML > DL > LLM 的关系
2. Token 换算（中文 × 1.5）
3. Temperature 的作用
4. Embedding 的用途
5. 余弦相似度范围（-1 到 1）
```

### 下一步

下一章：[机器学习基础](./02-machine-learning-basics.md)

---

**💡 学习建议**：
1. 把本章的术语抄一遍，每天看一遍
2. 用自己的话解释每个概念（能讲清楚才是真懂）
3. 跑一下 Embedding 的代码示例，加深理解

# 第七章：RAG 检索增强生成

> 💡 目标：掌握 RAG 原理和实现，这是前端最常做的 AI 项目类型
> 
> ⭐⭐⭐ 面试必考 + 项目必备

---

## 📖 什么是 RAG？

**定义**：Retrieval-Augmented Generation（检索增强生成）

**核心思想**：先检索相关文档，再让 AI 基于文档回答

```
传统 AI：
用户问 → AI 直接回答（可能胡说八道）

RAG：
用户问 → 检索相关文档 → 拼接到 Prompt → AI 基于文档回答（更准确）
```

**类比**：
```
传统 AI：开卷考试，但只能靠记忆
RAG：开卷考试 + 可以查资料（准确率高）
```

---

## 🏗️ RAG 架构详解

### 完整流程

```
┌─────────────────────────────────────────────────────────┐
│                    RAG 工作流程                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  准备阶段（离线）：                                      │
│  1. 收集文档（PDF、Markdown、网页等）                   │
│  2. 文档分块（Chunking）                                │
│  3. 向量化（Embedding）                                 │
│  4. 存入向量数据库                                      │
│                                                         │
│  查询阶段（在线）：                                      │
│  1. 用户提问                                            │
│  2. 问题向量化                                          │
│  3. 检索相似文档（余弦相似度）                          │
│  4. 拼接 Prompt：问题 + 相关文档                        │
│  5. 调用大模型生成答案                                  │
│  6. 返回答案 + 引用来源                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 代码实现（完整示例）

```javascript
// 技术栈：Node.js + OpenAI + Pinecone（向量数据库）

import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

// ==================== 准备阶段 ====================

// 1. 文档分块
function chunkDocument(text, chunkSize = 500, overlap = 50) {
  const chunks = [];
  let start = 0;
  
  while (start < text.length) {
    const end = start + chunkSize;
    const chunk = text.slice(start, end);
    chunks.push(chunk);
    start += chunkSize - overlap; // 重叠部分
  }
  
  return chunks;
}

// 2. 向量化 + 存入向量数据库
async function indexDocuments(documents) {
  const index = pinecone.index('my-documents');
  
  for (const doc of documents) {
    // 分块
    const chunks = chunkDocument(doc.content);
    
    for (let i = 0; i < chunks.length; i++) {
      // 向量化
      const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: chunks[i],
      });
      
      // 存入向量数据库
      await index.upsert([{
        id: `${doc.id}-chunk-${i}`,
        values: embedding.data[0].embedding,
        metadata: {
          content: chunks[i],
          source: doc.source,
          docId: doc.id,
        },
      }]);
    }
  }
}

// ==================== 查询阶段 ====================

// 3. 检索相关文档
async function retrieveRelevantDocs(query, topK = 3) {
  const index = pinecone.index('my-documents');
  
  // 问题向量化
  const embedding = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: query,
  });
  
  // 向量搜索
  const results = await index.query({
    vector: embedding.data[0].embedding,
    topK: topK,
    includeMetadata: true,
  });
  
  // 提取文档内容
  return results.matches.map(match => ({
    content: match.metadata.content,
    source: match.metadata.source,
    score: match.score,
  }));
}

// 4. 生成答案
async function generateAnswer(query, relevantDocs) {
  // 构建 Prompt
  const context = relevantDocs
    .map((doc, i) => `[文档${i + 1}] ${doc.content}`)
    .join('\n\n');
  
  const prompt = `请根据下面的文档回答问题。如果文档中没有答案，请说"根据提供的文档，无法回答这个问题"。

文档：
${context}

问题：${query}

请用中文回答，并注明引用来源。`;

  // 调用大模型
  const completion = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.7,
  });
  
  return {
    answer: completion.choices[0].message.content,
    sources: relevantDocs.map(doc => doc.source),
  };
}

// ==================== 完整流程 ====================

async function ragQuery(query) {
  // 1. 检索相关文档
  const relevantDocs = await retrieveRelevantDocs(query);
  
  // 2. 生成答案
  const result = await generateAnswer(query, relevantDocs);
  
  // 3. 返回结果
  return result;
}

// 使用示例
const answer = await ragQuery('公司的年假政策是什么？');
console.log(answer.answer);
console.log('来源:', answer.sources);
```

---

## 🔧 核心技术详解

### 1. 文档分块（Chunking）

**为什么需要分块**：
- 文档太长，超过模型 Context 限制
- 只检索相关片段，节省 Token
- 提高检索精度

**分块策略**：

```javascript
// 策略 1：固定大小分块
chunkDocument(text, chunkSize = 500, overlap = 50);

// 策略 2：按段落分块
function chunkByParagraph(text) {
  return text.split(/\n\n+/).filter(p => p.trim());
}

// 策略 3：按语义分块（高级）
// 使用 NLP 模型检测语义边界
```

**最佳实践**：
- Chunk Size：300-800 tokens
- Overlap：50-100 tokens（避免切断上下文）
- 按自然边界分块（段落、标题）

### 2. 向量化（Embedding）

**常用 Embedding 模型**：

| 模型 | 维度 | 价格 | 特点 |
|------|------|------|------|
| text-embedding-3-small | 1536 | $0.02/1M | 性价比高 |
| text-embedding-3-large | 3072 | $0.13/1M | 效果最好 |
| m3e-base | 768 | 免费 | 中文优化 |

**Embedding 调用示例**：

```javascript
const embedding = await openai.embeddings.create({
  model: 'text-embedding-3-small',
  input: '你好，世界',
});

console.log(embedding.data[0].embedding);
// [0.0023, -0.0045, 0.0067, ...]  // 1536 维向量
```

### 3. 向量数据库

**主流选择**：

| 数据库 | 类型 | 价格 | 特点 |
|--------|------|------|------|
| **Pinecone** | 托管 | 免费 + 付费 | 最简单 |
| **Weaviate** | 自托管/托管 | 免费 + 付费 | 功能多 |
| **Qdrant** | 自托管/托管 | 免费 + 付费 | 性能好 |
| **Milvus** | 自托管 | 免费 | 开源 |
| **Chroma** | 本地 | 免费 | 开发友好 |

**Pinecone 快速开始**：

```javascript
import { Pinecone } from '@pinecone-database/pinecone';

const pc = new Pinecone({ apiKey: 'xxx' });

// 创建索引
await pc.createIndex({
  name: 'my-index',
  dimension: 1536, // Embedding 维度
  metric: 'cosine', // 相似度算法
});

// 查询
const index = pc.index('my-index');
const results = await index.query({
  vector: [0.1, 0.2, ...], // 1536 维向量
  topK: 3,
  includeMetadata: true,
});
```

### 4. 相似度计算

**常用算法**：

```javascript
// 1. 余弦相似度（最常用）
function cosineSimilarity(a, b) {
  const dotProduct = a.reduce((sum, _, i) => sum + a[i] * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (normA * normB);
}

// 2. 欧氏距离
function euclideanDistance(a, b) {
  return Math.sqrt(a.reduce((sum, _, i) => sum + (a[i] - b[i]) ** 2, 0));
}

// 3. 点积
function dotProduct(a, b) {
  return a.reduce((sum, _, i) => sum + a[i] * b[i], 0);
}
```

**选择建议**：
- 余弦相似度：文本检索（推荐）
- 欧氏距离：聚类分析
- 点积：快速近似

---

## 📝 Prompt 设计技巧

### RAG Prompt 模板

```javascript
const prompt = `请根据下面的文档回答问题。

【要求】
1. 只基于文档内容回答，不要编造
2. 如果文档中没有答案，明确说"无法从文档中找到答案"
3. 回答要简洁，不超过 200 字
4. 注明引用来源

【文档】
${context}

【问题】
${query}

【回答】`;
```

### 带引用标注

```javascript
const prompt = `请根据文档回答问题，并在回答中标注引用来源。

格式要求：
- 使用 [1]、[2] 标注引用
- 最后列出参考文献

文档：
[1] ${doc1}
[2] ${doc2}
[3] ${doc3}

问题：${query}

回答：`;
```

---

## 🎯 实际项目案例

### 案例 1：公司内部知识库

**需求**：员工可以快速查询公司政策、流程、文档

**技术栈**：
- 前端：React + TypeScript
- 后端：Node.js + Express
- 向量库：Pinecone
- 大模型：GPT-4o

**架构**：
```
员工提问 → 前端 → 后端 → 检索文档 → 调用 GPT → 返回答案
                              ↓
                         Pinecone
```

**效果**：
- 准确率：90%+
- 响应时间：< 2 秒
- 成本：每次查询约 $0.01

### 案例 2：电商客服机器人

**需求**：自动回答商品、订单、售后问题

**特殊处理**：
```javascript
// 1. 多轮对话支持
const conversationHistory = [
  { role: 'user', content: '这个商品有货吗？' },
  { role: 'assistant', content: '有货的，亲~' },
  { role: 'user', content: '什么时候发货？' }, // 需要上下文
];

// 2. 意图识别 + RAG
async function handleQuery(query, history) {
  // 先识别意图
  const intent = await classifyIntent(query);
  
  if (intent === 'product_query') {
    // 查询商品知识库
    return ragQuery(query, 'product_docs');
  } else if (intent === 'order_query') {
    // 查询订单数据库
    return queryOrderDatabase(query);
  } else {
    // 通用问答
    return ragQuery(query, 'general_docs');
  }
}
```

### 案例 3：法律文档检索

**特殊需求**：
- 高准确率（不能胡说）
- 必须标注引用
- 支持复杂查询

**解决方案**：
```javascript
// 1. 严格模式 Prompt
const prompt = `你是一位法律助手。请严格基于提供的法条回答问题。

重要：
- 必须引用具体法条编号
- 不能推测或延伸
- 如果法条没有规定，明确说明

法条：
${legalDocuments}

问题：${query}

回答：`;

// 2. 置信度阈值
const results = await retrieveRelevantDocs(query);
const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;

if (avgScore < 0.7) {
  return '抱歉，没有找到足够相关的法条。建议咨询专业律师。';
}
```

---

## 🐛 常见问题和解决方案

### 问题 1：检索结果不相关

**原因**：
- 分块太大或太小
- Embedding 模型不合适
- 查询表述不清晰

**解决**：
```javascript
// 1. 调整分块大小
chunkDocument(text, chunkSize = 300, overlap = 50); // 减小 chunk

// 2. 换 Embedding 模型
model: 'text-embedding-3-large'; // 更好的模型

// 3. Query 重写
const rewrittenQuery = await rewriteQuery(query);
// "年假" → "公司员工年假政策规定"
```

### 问题 2：答案不准确（幻觉）

**原因**：
- Prompt 没有强调基于文档
- 模型温度太高
- 检索文档质量差

**解决**：
```javascript
// 1. 强化 Prompt
const prompt = `请严格基于下面的文档回答。如果文档中没有，就说不知道。`;

// 2. 降低温度
temperature: 0.3; // 更确定

// 3. 增加引用要求
请标注引用来源 [1]、[2]...
```

### 问题 3：响应太慢

**原因**：
- 检索文档太多
- Embedding 计算慢
- 网络延迟

**解决**：
```javascript
// 1. 减少检索数量
topK: 3; // 只检索 3 个最相关的

// 2. 缓存 Embedding
const cached = await cache.get(query);
if (cached) return cached;

// 3. 异步处理
const [docs, embedding] = await Promise.all([
  retrieveDocs(query),
  computeEmbedding(query),
]);
```

---

## 🎯 面试高频问题

### Q1: 什么是 RAG？为什么需要它？

**参考回答**：
```
RAG = Retrieval-Augmented Generation（检索增强生成）

核心思想：先检索相关文档，再让 AI 基于文档回答。

为什么需要：
1. 解决幻觉问题：AI 不会胡说八道
2. 提供最新信息：模型训练数据可能过时
3. 私有知识：公司内部文档模型不知道
4. 可追溯：可以标注引用来源

应用场景：
- 客服机器人（基于产品文档）
- 知识库问答（基于公司文档）
- 法律/医疗助手（基于专业文档）
```

### Q2: RAG 的完整流程是什么？

**参考回答**：
```
分为准备阶段和查询阶段：

准备阶段（离线）：
1. 收集文档
2. 文档分块（Chunking）
3. 向量化（Embedding）
4. 存入向量数据库

查询阶段（在线）：
1. 用户提问
2. 问题向量化
3. 检索相似文档（余弦相似度）
4. 拼接 Prompt：问题 + 相关文档
5. 调用大模型生成答案
6. 返回答案 + 引用来源
```

### Q3: 如何评估 RAG 系统的好坏？

**参考回答**：
```
我从几个维度评估：

1. 检索质量
   - 召回率：相关文档是否都被检索出来
   - 精度：检索的文档是否相关
   - 指标：MRR、NDCG

2. 生成质量
   - 准确性：答案是否正确
   - 忠实度：是否基于文档（不幻觉）
   - 人工评估 + 自动评估

3. 性能指标
   - 响应时间：< 2 秒
   - 并发能力
   - 成本 per query

4. 用户体验
   - 用户满意度
   - 问题解决率
   - 转人工率
```

### Q4: 你做过 RAG 相关项目吗？

**参考回答**：
```
我做过一个公司内部知识库项目：

需求：员工查询公司政策、流程

技术栈：
- Node.js + Express 后端
- React 前端
- Pinecone 向量数据库
- GPT-4o 生成答案

挑战：
1. 文档格式不统一（PDF、Word、Markdown）
   解决：统一转成 Markdown，再分块

2. 检索结果不相关
   解决：调整 chunk size 到 300，增加 overlap

3. AI 幻觉
   解决：Prompt 强调"只基于文档"，温度设为 0.3

效果：
- 准确率 90%+
- 日均 500+ 查询
- 减少 HR 重复咨询 70%
```

---

## 📝 自测题

**1. RAG 中为什么要文档分块？**

<details>
<summary>点击查看答案</summary>

1. 避免超过 Context 限制
2. 只检索相关片段，节省 Token
3. 提高检索精度
</details>

**2. 余弦相似度的范围是多少？**

<details>
<summary>点击查看答案</summary>

-1 到 1
1：完全相同
0：无关
-1：完全相反
</details>

**3. 如何减少 AI 幻觉？**

<details>
<summary>点击查看答案</summary>

1. Prompt 强调"只基于文档"
2. 降低 temperature（如 0.3）
3. 要求标注引用来源
4. 设置置信度阈值
</details>

**4. 向量数据库有哪些？**

<details>
<summary>点击查看答案</summary>

Pinecone、Weaviate、Qdrant、Milvus、Chroma
</details>

---

## 🏃 下一步

下一章：[Function Calling 和 Tool Use](./08-function-calling.md)

---

**💡 学习建议**：RAG 是前端最常做的 AI 项目类型，一定要动手实践！

**实战项目建议**：
1. 用 DeepSeek（免费）+ Chroma（本地）搭建一个简单的知识库
2. 收集你自己的笔记作为文档
3. 实现问答功能
4. 部署到 GitHub Pages + Vercel

这个项目可以写进简历，面试时有东西可聊！

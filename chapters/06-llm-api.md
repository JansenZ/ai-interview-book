# 第六章：调用大模型 API

> 💡 目标：掌握主流大模型 API 的调用方法，能快速集成到项目中
> 
> ⭐⭐⭐ 面试必考 + 实际工作必会

---

## 📖 主流大模型 API 对比

### 国际厂商

| 厂商 | 模型 | 价格（输入/输出） | Context | 特点 |
|------|------|------------------|---------|------|
| **OpenAI** | GPT-4o | $5 / $15 per 1M tokens | 128K | 最强通用能力 |
| **Anthropic** | Claude 3.5 Sonnet | $3 / $15 per 1M tokens | 200K | 长文本、代码强 |
| **Google** | Gemini Pro | $0.5 / $1.5 per 1M tokens | 128K | 多模态好 |
| **Meta** | Llama 3 | 开源 | 8K-128K | 可本地部署 |

### 国内厂商

| 厂商 | 模型 | 价格（输入/输出） | Context | 特点 |
|------|------|------------------|---------|------|
| **阿里** | 通义千问 Plus | ¥0.004 / ¥0.012 per 1K tokens | 32K | 中文好 |
| **百度** | 文心一言 4.0 | ¥0.012 / ¥0.012 per 1K tokens | 128K | 中文优化 |
| **腾讯** | 混元 | ¥0.001 / ¥0.001 per 1K tokens | 32K | 便宜 |
| **DeepSeek** | DeepSeek V2 | ¥0.001 / ¥0.002 per 1K tokens | 128K | 性价比之王 |

**价格对比（100 万 tokens）**：
```
GPT-4o:     $20  (约¥145)
Claude 3.5: $18  (约¥130)
通义千问：  ¥16  (约$2.2)
DeepSeek:   ¥3   (约$0.4)  ← 最便宜
```

---

## 🔑 API Key 获取

### OpenAI

1. 注册：https://platform.openai.com
2. 创建 API Key：https://platform.openai.com/api-keys
3. 格式：`sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

⚠️ 需要国外手机号 + 信用卡

### Anthropic (Claude)

1. 注册：https://console.anthropic.com
2. 创建 API Key：https://console.anthropic.com/settings/keys
3. 格式：`sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

⚠️ 需要国外手机号 + 信用卡

### 通义千问（阿里）

1. 注册：https://dashscope.console.aliyun.com
2. 创建 API Key：https://dashscope.console.aliyun.com/apiKey
3. 格式：`sk-xxxxxxxxxxxxxxxx`

✅ 国内可直接使用，有免费额度

### DeepSeek

1. 注册：https://platform.deepseek.com
2. 创建 API Key：https://platform.deepseek.com/api_keys
3. 格式：`sk-xxxxxxxxxxxxxxxx`

✅ 国内可使用，性价比最高

---

## 💻 调用示例代码

### 1. OpenAI GPT-4

```javascript
// 安装：npm install openai
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function chat() {
  const completion = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [
      { role: 'system', content: '你是一位专业的助手' },
      { role: 'user', content: '你好，请介绍一下自己' },
    ],
    temperature: 0.7,
    max_tokens: 1000,
  });

  console.log(completion.choices[0].message.content);
}

chat();
```

### 2. Anthropic Claude

```javascript
// 安装：npm install @anthropic-ai/sdk
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function chat() {
  const message = await anthropic.messages.create({
    model: 'claude-3-5-sonnet-20241022',
    max_tokens: 1024,
    messages: [
      { role: 'user', content: '你好，请介绍一下自己' },
    ],
  });

  console.log(message.content[0].text);
}

chat();
```

### 3. 通义千问（阿里）

```javascript
// 安装：npm install dashscope
import DashScope from 'dashscope';

DashScope.apiKey = process.env.DASHSCOPE_API_KEY;

async function chat() {
  const response = await DashScope.call({
    model: 'qwen-plus',
    messages: [
      { role: 'system', content: '你是一位专业的助手' },
      { role: 'user', content: '你好，请介绍一下自己' },
    ],
  });

  console.log(response.output.text);
}

chat();
```

### 4. DeepSeek

```javascript
// DeepSeek 兼容 OpenAI API 格式
// 安装：npm install openai
import OpenAI from 'openai';

const deepseek = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: 'https://api.deepseek.com', // 注意 baseURL
});

async function chat() {
  const completion = await deepseek.chat.completions.create({
    model: 'deepseek-chat',
    messages: [
      { role: 'user', content: '你好，请介绍一下自己' },
    ],
  });

  console.log(completion.choices[0].message.content);
}

chat();
```

---

## 🌊 流式输出（Streaming）

**作用**：打字机效果，用户体验更好

### OpenAI 流式输出

```javascript
async function chatStream() {
  const stream = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: '写一个故事' }],
    stream: true, // 开启流式
  });

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content || '';
    process.stdout.write(content); // 逐字输出
  }
}

chatStream();
```

### React 中实现打字机效果

```jsx
import { useState } from 'react';

function ChatComponent() {
  const [response, setResponse] = useState('');

  async function sendMessage() {
    const stream = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: '你好' }],
      stream: true,
    });

    let fullResponse = '';
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      fullResponse += content;
      setResponse(fullResponse); // 实时更新 UI
    }
  }

  return (
    <div>
      <button onClick={sendMessage}>发送</button>
      <div>{response}</div>
    </div>
  );
}
```

---

## 📊 Token 计算和成本控制

### Token 计算规则

```
英文：1 个单词 ≈ 1.3 tokens
中文：1 个汉字 ≈ 1.5 tokens
代码：1 个字符 ≈ 1 token

示例：
"Hello, world!" ≈ 4 tokens
"你好，世界！" ≈ 6 tokens
```

### 在线计算工具

- OpenAI Tokenizer: https://platform.openai.com/tokenizer
- 通义千问 Token 计算：https://help.aliyun.com/document_detail/2423588.html

### 成本估算示例

```javascript
// 计算一次对话的成本
function calculateCost(inputTokens, outputTokens, model) {
  const prices = {
    'gpt-4o': { input: 5, output: 15 }, // $ per 1M tokens
    'claude-3-5-sonnet': { input: 3, output: 15 },
    'qwen-plus': { input: 0.004, output: 0.012 }, // ¥ per 1K tokens
    'deepseek-chat': { input: 0.001, output: 0.002 },
  };

  const price = prices[model];
  const inputCost = (inputTokens / 1_000_000) * price.input;
  const outputCost = (outputTokens / 1_000_000) * price.output;

  return { inputCost, outputCost, total: inputCost + outputCost };
}

// 示例：1000 tokens 输入，500 tokens 输出
console.log(calculateCost(1000, 500, 'gpt-4o'));
// { inputCost: 0.005, outputCost: 0.0075, total: 0.0125 } $

console.log(calculateCost(1000, 500, 'deepseek-chat'));
// { inputCost: 0.000001, outputCost: 0.000001, total: 0.000002 } $
```

### 省钱技巧

1. **选择合适的模型**
   - 简单任务用小模型（GPT-3.5-Turbo、DeepSeek）
   - 复杂任务用大模型（GPT-4、Claude）

2. **优化 Prompt**
   - 减少不必要的上下文
   - 明确要求，减少重复调用

3. **缓存结果**
   - 相同的问题直接返回缓存
   - 使用向量数据库做语义缓存

4. **设置 Token 限制**
   ```javascript
   const completion = await openai.chat.completions.create({
     model: 'gpt-4o',
     messages: [...],
     max_tokens: 500, // 限制输出长度
   });
   ```

---

## 🔒 安全最佳实践

### 1. 不要硬编码 API Key

```javascript
// ❌ 错误：硬编码
const apiKey = 'sk-xxxxxxxxxxxxxxxx';

// ✅ 正确：环境变量
const apiKey = process.env.OPENAI_API_KEY;
```

### 2. 前端不要直接调用 API

```javascript
// ❌ 错误：前端直接调用（API Key 会暴露）
fetch('https://api.openai.com/v1/chat/completions', {
  headers: {
    'Authorization': `Bearer ${API_KEY}`, // 暴露！
  }
});

// ✅ 正确：通过后端代理
// 前端
fetch('/api/chat', {
  method: 'POST',
  body: JSON.stringify({ message: '你好' }),
});

// 后端（Node.js）
app.post('/api/chat', async (req, res) => {
  const completion = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: req.body.message }],
  });
  res.json({ response: completion.choices[0].message.content });
});
```

### 3. 设置 Rate Limit

```javascript
// 使用 express-rate-limit
import rateLimit from 'express-rate-limit';

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 分钟
  max: 100, // 最多 100 次请求
  message: '请求太频繁，请稍后再试',
});

app.use('/api/chat', limiter);
```

### 4. 监控和告警

```javascript
// 记录每次调用的 token 消耗
async function trackUsage(model, usage) {
  await db.usage.create({
    model,
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens,
    timestamp: new Date(),
  });

  // 如果超过阈值，发送告警
  const dailyUsage = await db.usage.getDailyTotal();
  if (dailyUsage.cost > 100) {
    await sendAlert('今日 API 消费超过$100');
  }
}
```

---

## 🎯 面试高频问题

### Q1: 你用过哪些大模型 API？

**参考回答**：
```
我主要用过以下几个：

1. OpenAI GPT-4：能力最强，但价格贵，适合复杂任务
2. Claude 3.5：长文本处理很好，代码能力强
3. 通义千问：中文效果好，国内访问快
4. DeepSeek：性价比最高，适合预算有限的项目

选择建议：
- 追求效果：GPT-4 或 Claude
- 预算有限：DeepSeek 或通义千问
- 国内部署：通义千问或 DeepSeek
```

### Q2: 如何控制 API 调用成本？

**参考回答**：
```
我从几个方面控制成本：

1. 模型选择：简单任务用小模型，复杂任务用大模型
2. Prompt 优化：减少不必要的上下文，明确要求
3. 结果缓存：相同问题直接返回缓存
4. Token 限制：设置 max_tokens 防止输出过长
5. 监控告警：设置每日预算，超支自动告警

实际项目中，通过这些方法，我把成本降低了 60%+。
```

### Q3: 前端如何安全地调用大模型 API？

**参考回答**：
```
前端不应该直接调用大模型 API，因为会暴露 API Key。

正确做法：
1. 前端发送请求到后端
2. 后端验证用户身份和权限
3. 后端调用大模型 API
4. 后端返回结果给前端

额外安全措施：
- Rate Limit 限制调用频率
- 记录日志便于审计
- 设置每日预算和告警
- 敏感信息过滤
```

---

## 📝 自测题

**1. OpenAI API 的 baseURL 是什么？**

<details>
<summary>点击查看答案</summary>

`https://api.openai.com/v1`
</details>

**2. 如何开启流式输出？**

<details>
<summary>点击查看答案</summary>

设置 `stream: true`，然后用 `for await...of` 遍历响应流。
</details>

**3. 1000 个中文字大约多少 tokens？**

<details>
<summary>点击查看答案</summary>

约 1500 tokens（1 个汉字 ≈ 1.5 tokens）
</details>

**4. 为什么前端不能直接调用大模型 API？**

<details>
<summary>点击查看答案</summary>

会暴露 API Key，导致：
1. Key 被盗用，产生高额费用
2. 无法控制调用权限
3. 无法做 Rate Limit

正确做法是通过后端代理。
</details>

---

## 🏃 下一步

下一章：[RAG 检索增强生成](./07-rag.md)

---

**💡 学习建议**：这章的内容一定要动手实践！
1. 注册一个 DeepSeek 账号（有免费额度）
2. 用 Node.js 写一个简单的聊天程序
3. 尝试流式输出
4. 计算一下你的调用成本

实践一遍比看十遍都有效！

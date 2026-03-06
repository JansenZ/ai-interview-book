# 第九章：AI Agent（智能体）

## 9.1 什么是 AI Agent

**AI Agent（智能体）** 是指能够感知环境、做出决策、执行动作以实现目标的智能系统。

### 核心公式

```
Agent = 大模型 + 感知 + 规划 + 记忆 + 工具使用
```

### Agent 与传统 AI 的区别

| 对比项 | 传统 AI/Chatbot | AI Agent |
|--------|----------------|----------|
| **目标** | 回答问题 | 完成任务 |
| **交互** | 单轮/多轮对话 | 多步骤执行 |
| **能力** | 仅语言生成 | 可调用工具/API |
| **记忆** | 有限上下文 | 长期记忆存储 |
| **自主性** | 被动响应 | 主动规划执行 |

---

## 9.2 Agent 的核心组件

### 9.2.1 大脑（Brain）

**大语言模型**作为核心决策器：
- 理解任务
- 制定计划
- 做出决策
- 生成响应

**模型选择考虑**：
- **能力**：推理、代码、工具调用能力
- **上下文长度**：长任务需要大 context
- **成本**：Token 价格
- **延迟**：实时交互需要低延迟

---

### 9.2.2 感知（Perception）

Agent 接收外界信息的能力：

1. **文本输入**：用户指令、文档内容
2. **视觉输入**：图像识别、OCR
3. **听觉输入**：语音识别
4. **多模态**：综合多种输入

```python
# 多模态输入示例
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "这张图片里有什么？"},
        {"type": "image_url", "image_url": {"url": "image.jpg"}}
    ]}
]
```

---

### 9.2.3 规划（Planning）

将复杂任务分解为可执行的步骤。

#### 任务分解（Task Decomposition）

```
任务：帮我研究新能源汽车市场并写一份报告

分解：
1. 搜索新能源汽车市场最新数据
2. 分析主要品牌市场份额
3. 查找技术发展趋势
4. 整理政策环境信息
5. 综合信息撰写报告
```

#### 规划方法

**1. 链式思考（Chain of Thought, CoT）**
```
问题：小明有 5 个苹果，吃了 2 个，又买了 3 个，现在有几个？

思考过程：
- 初始：5 个苹果
- 吃了 2 个：5 - 2 = 3 个
- 又买 3 个：3 + 3 = 6 个
- 答案：6 个
```

**2. 思维树（Tree of Thoughts, ToT）**
```
        根问题
       /  |  \
     方案 1 方案 2 方案 3
      |      |      |
    评估   评估   评估
      \      |      /
        最优方案
```

**3. 反思（Reflection）**
```python
def solve_with_reflection(task):
    solution = llm.generate(task)
    feedback = llm.evaluate(solution, task)
    if not feedback.is_satisfactory:
        solution = llm.revise(solution, feedback)
    return solution
```

---

### 9.2.4 记忆（Memory）

Agent 存储和检索信息的能力。

#### 记忆类型

| 类型 | 特点 | 实现方式 |
|------|------|---------|
| **短期记忆** | 当前会话上下文 | LLM Context Window |
| **长期记忆** | 跨会话持久化 | 向量数据库 |
| **程序记忆** | 技能和习惯 | Fine-tuning / Prompt |

#### 向量数据库实现

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 初始化向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./memory")

# 添加记忆
vectorstore.add_texts(["今天和用户讨论了 AI Agent 架构"])

# 检索相关记忆
relevant_memories = vectorstore.similarity_search("Agent 架构", k=3)
```

#### 记忆管理策略

1. **重要性评分**：给记忆打分，保留重要的
2. **时间衰减**：近期记忆权重更高
3. **相关性检索**：根据当前任务检索相关记忆
4. **定期整理**：合并、删除冗余记忆

---

### 9.2.5 工具使用（Tool Use）

Agent 调用外部工具/API 扩展能力。

#### Function Calling

```python
from openai import OpenAI

client = OpenAI()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索网络信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    }
]

# 调用模型
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools
)

# 处理工具调用
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.function.name == "get_weather":
        args = json.loads(tool_call.function.arguments)
        weather = get_weather(args["city"])  # 执行实际函数
```

#### 常见工具类型

| 工具类型 | 示例 |
|---------|------|
| **搜索** | Google Search、Bing API |
| **计算** | Python 解释器、Wolfram Alpha |
| **数据库** | SQL 查询、向量检索 |
| **API** | 天气、股票、新闻 API |
| **文件** | 读写文件、PDF 解析 |
| **代码** | 执行代码、调试 |
| **浏览器** | 网页爬取、自动化操作 |

---

## 9.3 Agent 架构模式

### 9.3.1 ReAct（Reason + Act）

**核心思想**：交替进行推理和行动。

```
Thought: 我需要查找北京今天的天气
Action: get_weather("北京")
Observation: 晴，25°C
Thought: 现在我有了天气信息，可以回答用户了
Answer: 北京今天晴朗，气温 25°C
```

**实现框架**：
```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=[search_tool, calculator_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("北京今天天气如何？气温是多少华氏度？")
```

---

### 9.3.2 Plan-and-Execute

**核心思想**：先制定完整计划，再逐步执行。

```
步骤 1: 制定计划
  - 目标：写一份市场分析报告
  - 步骤：搜索数据 → 分析趋势 → 撰写报告

步骤 2: 执行计划
  - 执行步骤 1: 搜索市场数据 ✓
  - 执行步骤 2: 分析趋势 ✓
  - 执行步骤 3: 撰写报告 ✓

步骤 3: 输出结果
```

**适用场景**：
- 复杂多步骤任务
- 需要全局规划
- 步骤间有依赖关系

---

### 9.3.3 Reflexion

**核心思想**：通过反思改进表现。

```python
def reflexion_agent(task, max_iterations=3):
    trajectory = []
    
    for i in range(max_iterations):
        # 执行
        action, output = agent.act(task, trajectory)
        trajectory.append((action, output))
        
        # 评估
        feedback = agent.reflect(task, trajectory)
        
        if feedback.is_success:
            return output
        
        # 根据反思调整
        task = agent.revise_task(task, feedback)
    
    return output
```

**适用场景**：
- 需要试错的任务
- 有明确评估标准
- 可以迭代改进

---

### 9.3.4 Multi-Agent（多智能体）

**核心思想**：多个 Agent 协作完成复杂任务。

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  研究员 Agent  │ ──→ │  分析师 Agent  │ ──→ │  作家 Agent   │
│  (收集信息)   │     │  (分析数据)   │     │  (撰写报告)   │
└─────────────┘     └─────────────┘     └─────────────┘
                           ↓
                    ┌─────────────┐
                    │  审核 Agent   │
                    │  (质量检查)   │
                    └─────────────┘
```

**角色分配**：
- **研究员**：信息收集
- **分析师**：数据处理
- **作家**：内容创作
- **审核员**：质量把关
- **协调员**：任务分配

---

## 9.4 Agent 开发框架

### 9.4.1 LangChain

**特点**：功能全面、生态丰富

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage

# 定义工具
tools = [search_tool, calculator_tool, database_tool]

# 配置记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建 Agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 运行
result = agent_executor.invoke({"input": "帮我分析这家公司的财务状况"})
```

**核心组件**：
- **Chains**：任务链
- **Agents**：智能体
- **Tools**：工具集
- **Memory**：记忆
- **VectorStores**：向量存储

---

### 9.4.2 AutoGen

**特点**：微软出品，多 Agent 协作强

```python
from autogen import ConversableAgent, UserProxyAgent

# 创建 Agent
researcher = ConversableAgent(
    name="Researcher",
    system_message="你是研究专家，负责收集信息",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

analyst = ConversableAgent(
    name="Analyst",
    system_message="你是数据分析师",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "coding"}
)

# 启动对话
user_proxy.initiate_chat(
    researcher,
    message="帮我研究 AI 行业发展趋势",
    max_turns=5
)
```

**优势**：
- 多 Agent 对话
- 代码执行
- 灵活的人机协作

---

### 9.4.3 CrewAI

**特点**：角色驱动，流程清晰

```python
from crewai import Agent, Task, Crew

# 定义角色
researcher = Agent(
    role="高级研究员",
    goal="深入调研指定主题",
    backstory="你是经验丰富的行业研究员",
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role="内容作家",
    goal="撰写高质量报告",
    backstory="你是专业的技术作家",
    verbose=True
)

# 定义任务
task1 = Task(
    description="调研 AI Agent 市场现状",
    agent=researcher,
    expected_output="一份市场调研摘要"
)

task2 = Task(
    description="根据调研结果撰写报告",
    agent=writer,
    expected_output="一份完整的行业报告"
)

# 组建团队
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2
)

result = crew.kickoff()
```

---

### 9.4.4 LlamaIndex

**特点**：数据连接强，RAG 友好

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.agent import OpenAIAgent

# 加载数据
documents = SimpleDirectoryReader("./data").load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 创建 Agent
agent = OpenAIAgent.from_tools(
    tool_retriever=index.as_retriever(similarity_top_k=3),
    llm=llm,
    verbose=True
)

response = agent.chat("根据文档内容，总结一下主要观点")
```

---

## 9.5 Agent 实战案例

### 案例 1：智能客服 Agent

**需求**：
- 自动回答用户问题
- 查询订单状态
- 处理退换货请求
- 复杂问题转人工

**架构**：
```
用户输入 → 意图识别 → 路由分发
                    ├── 常见问题 → 知识库检索 → 回答
                    ├── 订单查询 → 订单 API → 返回状态
                    ├── 退换货 → 流程引导 → 生成工单
                    └── 复杂问题 → 转人工客服
```

**实现**：
```python
class CustomerServiceAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.tools = [
            order_lookup_tool,
            refund_tool,
            knowledge_base_tool
        ]
        self.memory = ConversationBufferMemory()
        
    def handle_request(self, user_input):
        # 意图识别
        intent = self.classify_intent(user_input)
        
        # 根据意图选择处理流程
        if intent == "order_inquiry":
            return self.handle_order_inquiry(user_input)
        elif intent == "refund_request":
            return self.handle_refund(user_input)
        elif intent == "general_question":
            return self.handle_general_question(user_input)
        else:
            return self.transfer_to_human(user_input)
```

---

### 案例 2：数据分析 Agent

**需求**：
- 连接数据库查询数据
- 执行统计分析
- 生成可视化图表
- 输出分析报告

**工具集**：
```python
tools = [
    SqlTool(),           # SQL 查询
    PandasTool(),        # 数据处理
    MatplotlibTool(),    # 图表生成
    StatisticsTool(),    # 统计分析
    ReportGeneratorTool() # 报告生成
]
```

**工作流**：
```
1. 理解分析需求
2. 生成 SQL 查询数据
3. 使用 Pandas 处理数据
4. 生成可视化图表
5. 撰写分析报告
```

---

### 案例 3：个人助理 Agent

**功能**：
- 日程管理
- 邮件处理
- 信息检索
- 任务提醒

**记忆系统**：
```python
class PersonalAssistantMemory:
    def __init__(self):
        self.short_term = []  # 短期记忆
        self.long_term = VectorStore()  # 长期记忆
        self.preferences = {}  # 用户偏好
        
    def add_memory(self, content, importance=0.5):
        # 存储到短期记忆
        self.short_term.append({
            "content": content,
            "timestamp": datetime.now(),
            "importance": importance
        })
        
        # 重要记忆存入长期记忆
        if importance > 0.7:
            self.long_term.add(content)
    
    def get_relevant_memories(self, query):
        return self.long_term.search(query, k=5)
```

---

## 9.6 Agent 评估

### 评估维度

| 维度 | 指标 | 说明 |
|------|------|------|
| **任务完成度** | 成功率 | 是否完成目标 |
| **效率** | 步骤数/时间 | 完成任务的代价 |
| **准确性** | 错误率 | 决策/执行错误比例 |
| **鲁棒性** | 异常处理 | 面对意外的表现 |
| **安全性** | 有害行为数 | 是否产生危险输出 |

### 评估方法

**1. 基准测试**
```python
def evaluate_agent(agent, benchmark_tasks):
    results = []
    for task in benchmark_tasks:
        result = agent.execute(task)
        score = task.evaluate(result)
        results.append(score)
    return np.mean(results)
```

**2. A/B 测试**
- 对比不同配置
- 用户反馈收集
- 关键指标监控

**3. 人工评估**
- 专家评审
- 用户满意度
- 案例研究

---

## 9.7 挑战与解决方案

### 挑战 1：幻觉（Hallucination）

**问题**：Agent 生成错误或虚构信息

**解决方案**：
1. **检索增强**：基于事实数据生成
2. **工具验证**：用工具核实关键信息
3. **自我反思**：让模型检查自己的输出
4. **多轮确认**：重要信息多次验证

```python
def verify_with_tools(agent, claim):
    # 使用搜索工具验证
    search_result = agent.search(claim)
    if not supports_claim(search_result):
        return agent.revise(claim, search_result)
    return claim
```

---

### 挑战 2：无限循环

**问题**：Agent 陷入重复动作

**解决方案**：
1. **最大迭代次数**：设置上限
2. **状态检测**：检测重复状态
3. **超时机制**：时间限制
4. **人工介入**：异常情况转人工

```python
def execute_with_limits(agent, task, max_steps=10):
    history = []
    for i in range(max_steps):
        action = agent.decide(task, history)
        
        # 检测循环
        if action in history[-3:]:
            return agent.fallback(task)
        
        history.append(action)
        result = agent.execute(action)
        
        if result.is_done:
            return result
    
    return agent.timeout_handler(task)
```

---

### 挑战 3：安全性

**问题**：Agent 可能执行危险操作

**解决方案**：
1. **权限控制**：限制可执行操作
2. **沙箱环境**：隔离执行
3. **人工审核**：敏感操作需确认
4. **审计日志**：记录所有操作

```python
class SafeAgent:
    def __init__(self):
        self.allowed_tools = {"search", "read_file"}
        self.blocked_tools = {"delete", "execute_shell"}
        
    def execute(self, action):
        if action.tool in self.blocked_tools:
            raise SecurityError(f"Blocked tool: {action.tool}")
        if action.tool not in self.allowed_tools:
            return self.request_approval(action)
        return self.safe_execute(action)
```

---

### 挑战 4：成本控制

**问题**：Agent 多轮调用成本高

**解决方案**：
1. **模型分层**：简单任务用小模型
2. **缓存结果**：避免重复计算
3. **优化 Prompt**：减少 Token 消耗
4. **批量处理**：合并多个请求

```python
class CostOptimizedAgent:
    def __init__(self):
        self.simple_llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.complex_llm = ChatOpenAI(model="gpt-4")
        self.cache = {}
        
    def decide(self, task):
        # 检查缓存
        if task in self.cache:
            return self.cache[task]
        
        # 简单任务用小模型
        if self.is_simple(task):
            result = self.simple_llm.invoke(task)
        else:
            result = self.complex_llm.invoke(task)
        
        self.cache[task] = result
        return result
```

---

## 9.8 最佳实践

### ✅ 推荐做法

1. **明确边界**：清楚定义 Agent 能做什么
2. **渐进式开发**：从简单任务开始
3. **充分测试**：覆盖各种场景
4. **监控日志**：记录所有决策和行动
5. **用户反馈**：持续收集改进
6. **安全优先**：设置安全护栏
7. **成本意识**：优化 Token 使用

### ❌ 避免的坑

1. **过度设计**：一开始就搞复杂架构
2. **忽视评估**：只开发不测试
3. **无限信任**：完全相信模型输出
4. **忽略成本**：不考虑 Token 消耗
5. **缺少监控**：上线后无法追踪

---

## 9.9 面试真题

### 真题 1：解释 ReAct 框架的工作原理

**参考答案**：

ReAct（Reason + Act）是一种结合推理和行动的 Agent 框架。

**工作流程**：
1. **Thought（思考）**：分析当前状态，决定下一步
2. **Action（行动）**：执行具体操作（调用工具）
3. **Observation（观察）**：获取行动结果
4. **循环**：重复直到任务完成

**示例**：
```
Thought: 用户想知道北京的天气，我需要调用天气 API
Action: get_weather("北京")
Observation: 晴，25°C
Thought: 我有了天气信息，现在可以回答用户了
Answer: 北京今天晴朗，气温 25°C
```

**优势**：
- 透明可解释：思考过程可见
- 灵活：可调用各种工具
- 准确：基于实际数据回答

---

### 真题 2：如何设计一个多 Agent 协作系统？

**参考答案**：

设计多 Agent 系统需要考虑：

**1. 角色划分**
- 根据任务类型定义不同角色
- 每个角色有明确的职责和能力
- 例如：研究员、分析师、作家、审核员

**2. 通信机制**
- 共享上下文/黑板
- 消息传递
- 发布 - 订阅模式

**3. 协调策略**
- 顺序执行：A 完成后 B 开始
- 并行执行：多个 Agent 同时工作
- 混合模式：部分并行 + 部分顺序

**4. 冲突解决**
- 投票机制
- 仲裁 Agent
- 人工介入

**5. 实现框架**
- AutoGen：支持多 Agent 对话
- CrewAI：角色驱动流程
- 自定义：基于消息队列

**示例架构**：
```
协调器 → 分配任务给各 Agent
         ↓
    研究员 → 收集信息
    分析师 → 处理数据
    作家 → 生成内容
         ↓
    审核员 → 质量检查
         ↓
    输出最终结果
```

---

### 真题 3：Agent 出现幻觉怎么办？

**参考答案**：

**原因分析**：
- 模型本身的知识局限
- 过度自信生成
- 缺乏事实核查

**解决方案**：

1. **检索增强（RAG）**
   - 从可靠数据源检索信息
   - 基于检索结果生成回答
   - 引用来源增加可信度

2. **工具验证**
   - 关键信息用工具核实
   - 搜索 API 验证事实
   - 数据库查询确认

3. **自我反思**
   - 让模型检查自己的输出
   - 识别不确定内容
   - 标注置信度

4. **多轮确认**
   - 重要信息多次验证
   - 交叉检查不同来源
   - 人工审核关键环节

5. **系统提示**
   - 明确告知模型要诚实
   - 不知道就说不知道
   - 不要编造信息

**实践建议**：
- 对关键领域（医疗、法律）更严格
- 建立事实核查流程
- 记录幻觉案例持续改进

---

## 本章小结

1. **Agent = 大模型 + 感知 + 规划 + 记忆 + 工具**
2. **核心组件**：大脑、感知、规划、记忆、工具使用
3. **架构模式**：ReAct、Plan-and-Execute、Reflexion、Multi-Agent
4. **开发框架**：LangChain、AutoGen、CrewAI、LlamaIndex
5. **关键挑战**：幻觉、循环、安全、成本
6. **评估维度**：任务完成度、效率、准确性、鲁棒性、安全性

---

## 下一章预告

第十章将是本书的**最后一章**，内容包括：
- AI 行业趋势与展望
- 职业发展建议
- 学习路线图
- 面试准备清单
- 结语

让我们一起完成这本 AI 面试指南！🎯

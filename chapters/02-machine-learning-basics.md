# 第二章：机器学习基础

> 💡 **本章目标**：理解机器学习的核心概念，能解释常见算法的原理和应用场景
> 
> ⏱️ **预计时间**：3-4 小时
> 
> 📌 **面试重要度**：⭐⭐⭐⭐（高频考点）

---

## 2.1 机器学习 vs 传统编程

这是理解机器学习的第一步，必须搞懂。

### 核心区别

```
传统编程：
输入（数据） + 规则（程序） → 输出（答案）

机器学习：
输入（数据） + 输出（答案） → 规则（模型）
```

### 具体例子：垃圾邮件检测

**传统编程方式**：
```javascript
// 人工写规则
function isSpam(email) {
  // 规则 1：包含敏感词
  if (email.content.includes('中奖')) return true;
  if (email.content.includes('免费')) return true;
  if (email.content.includes('转账')) return true;
  
  // 规则 2：发件人可疑
  if (email.sender.includes('unknown')) return true;
  
  // 规则 3：多个收件人
  if (email.to.length > 10) return true;
  
  return false;
}

// 问题：
// 1. 规则要人工想，容易遗漏
// 2. 垃圾邮件不断演化，规则要不断更新
// 3. 复杂模式无法用简单规则描述
```

**机器学习方式**：
```javascript
// 1. 准备训练数据
const trainingData = [
  { email: email1, isSpam: true },
  { email: email2, isSpam: false },
  { email: email3, isSpam: true },
  // ... 10000 封已标注的邮件
];

// 2. 训练模型
const model = trainClassifier(trainingData);

// 3. 使用模型预测新邮件
const result = model.predict(newEmail);
// 输出：{ isSpam: true, confidence: 0.95 }

// 优势：
// 1. 模型自己学习规律，不需要人工写规则
// 2. 能发现人想不到的模式
// 3. 新数据可以继续训练，自动适应
```

### 类比：教小孩认动物

```
传统编程：
告诉小孩规则：
- 有尖耳朵、胡须、尾巴的是猫
- 汪汪叫的是狗
- 会飞的是鸟

问题：
- 规则太死板（没尾巴的猫就不认识了？）
- 复杂情况无法描述（什么是"像猫"？）

机器学习：
给小孩看 100 张猫的照片、100 张狗的照片
小孩自己总结规律

优势：
- 能识别各种姿态的猫
- 能处理模糊情况（这只动物有点像猫又有点像狗）
```

### 什么时候用机器学习？

**适合 ML 的场景**：
1. 规则复杂，难以人工描述（图像识别、语音识别）
2. 规则不断变化（垃圾邮件、欺诈检测）
3. 需要从数据中发现模式（推荐系统、用户分群）
4. 有大量标注数据可用

**不适合 ML 的场景**：
1. 规则简单明确（计算器、排序）
2. 需要 100% 准确（银行转账、航空控制）
3. 没有训练数据
4. 需要完全可解释（法律判决、医疗诊断的部分场景）

---

## 2.2 监督学习、无监督学习、强化学习

机器学习的三大类型，面试必考。

### 监督学习（Supervised Learning）

**定义**：训练数据有标签（正确答案）。

**核心思想**：从"输入 - 输出"对中学习映射关系。

**类比**：
```
老师教学生做题：
- 给学生题目 + 答案（训练数据）
- 学生做题，老师批改（训练过程）
- 学生学会了解题方法（模型）
- 考试做新题（推理）
```

**常见任务**：

1. **分类（Classification）**：预测类别
```
输入：邮件内容
输出：垃圾邮件 or 正常邮件

输入：图片
输出：猫 or 狗 or 汽车

输入：用户行为
输出：会购买 or 不会购买
```

2. **回归（Regression）**：预测数值
```
输入：房屋特征（面积、位置、房龄）
输出：房价（¥500 万）

输入：广告投入
输出：销售额（¥100 万）

输入：历史股价
输出：明天股价
```

**常用算法**：
- 线性回归（Linear Regression）
- 逻辑回归（Logistic Regression）
- 决策树（Decision Tree）
- 随机森林（Random Forest）
- 支持向量机（SVM）
- 神经网络（Neural Network）

**代码示例**：
```javascript
// 使用 TensorFlow.js 训练一个分类模型
import * as tf from '@tensorflow/tfjs';

// 1. 准备数据
const trainingData = tf.tensor2d([
  [0, 0],  // 输入 1
  [0, 1],  // 输入 2
  [1, 0],  // 输入 3
  [1, 1],  // 输入 4
]);

const labels = tf.tensor2d([
  [0],  // 输出 1（异或：不同为 1）
  [1],  // 输出 2
  [1],  // 输出 3
  [0],  // 输出 4
]);

// 2. 定义模型
const model = tf.sequential();
model.add(tf.layers.dense({ units: 4, inputShape: [2], activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// 3. 编译模型
model.compile({ 
  optimizer: tf.train.adam(0.1),
  loss: 'binaryCrossentropy',
});

// 4. 训练
await model.fit(trainingData, labels, {
  epochs: 100,  // 训练 100 轮
});

// 5. 预测
const result = model.predict(tf.tensor2d([[0, 1]]));
console.log(result.dataSync()); // 输出接近 1（正确）
```

### 无监督学习（Unsupervised Learning）

**定义**：训练数据没有标签，让模型自己找规律。

**核心思想**：发现数据的内在结构。

**类比**：
```
给小孩一堆积木，不告诉他怎么玩
小孩自己发现：
- 可以按颜色分类
- 可以按大小分类
- 可以搭成房子
```

**常见任务**：

1. **聚类（Clustering）**：自动分组
```
场景：用户分群
输入：10000 个用户的购买行为
输出：
  - 群组 1：价格敏感型（喜欢打折）
  - 群组 2：品质导向型（买贵的）
  - 群组 3：冲动消费型（经常买不需要的）

场景：新闻分类
输入：1000 篇新闻文章
输出：
  - 群组 1：政治新闻
  - 群组 2：体育新闻
  - 群组 3：娱乐新闻
```

2. **降维（Dimensionality Reduction）**：压缩数据
```
场景：数据可视化
输入：100 维的用户特征
输出：2 维（可以在平面图上展示）

场景：特征压缩
输入：1000x1000 像素的图片（100 万维）
输出：100 维的特征向量（保留关键信息）
```

3. **关联规则（Association）**：发现关联
```
场景：购物篮分析
发现：买尿布的人经常买啤酒
应用：超市把尿布和啤酒放一起

场景：推荐系统
发现：看 A 电影的人经常看 B 电影
应用：推荐 B 电影给看 A 电影的人
```

**常用算法**：
- K-Means 聚类
- 层次聚类
- DBSCAN
- PCA（主成分分析）
- 自编码器（Autoencoder）

**代码示例**：
```javascript
// K-Means 聚类示例（伪代码）
const data = [
  [1, 2],   // 用户 1：月消费 1000，访问 2 次
  [1, 3],   // 用户 2
  [5, 8],   // 用户 3：月消费 5000，访问 8 次
  [6, 9],   // 用户 4
  [1, 1],   // 用户 5
  [5, 7],   // 用户 6
];

// 聚类：分成 2 组
const clusters = kMeans(data, k=2);

// 结果：
// 群组 0（低价值用户）：用户 1、2、5
// 群组 1（高价值用户）：用户 3、4、6

// 应用：
// - 对群组 0：发优惠券，刺激消费
// - 对群组 1：提供 VIP 服务，保持忠诚
```

### 强化学习（Reinforcement Learning）

**定义**：通过与环境互动，根据奖励/惩罚学习。

**核心思想**：试错学习，最大化累积奖励。

**类比**：
```
训练小狗：
- 小狗做对了（坐下），给奖励（零食）
- 小狗做错了（乱跑），给惩罚（批评）
- 小狗逐渐学会：坐下=有好吃的

关键：
- 不是告诉小狗"怎么坐下"
- 而是让它自己尝试，根据反馈调整
```

**核心概念**：
```
Agent（智能体）：学习的主体
Environment（环境）：智能体交互的对象
Action（动作）：智能体可以做的行为
State（状态）：当前的情况
Reward（奖励）：动作的反馈（正/负）
Policy（策略）：什么情况下做什么动作
```

**工作流程**：
```
1. 智能体观察当前状态
2. 根据策略选择动作
3. 执行动作
4. 获得奖励和新状态
5. 根据奖励调整策略
6. 重复 1-5，直到策略最优
```

**应用场景**：
- 游戏 AI（AlphaGo、Dota 2）
- 机器人控制
- 自动驾驶
- 资源调度（服务器、交通）
- 推荐系统（用户点击=奖励）

**经典案例：AlphaGo**：
```
Agent: AlphaGo 程序
Environment: 围棋棋盘
Action: 落子位置
State: 当前棋局
Reward: 赢=+1，输=-1

学习过程：
1. 自己和自己下棋（自我对弈）
2. 赢了的学习，输了的改进
3. 下了数百万盘后，超越人类
```

### 三种学习类型对比

| 对比项 | 监督学习 | 无监督学习 | 强化学习 |
|--------|----------|------------|----------|
| **数据** | 有标签 | 无标签 | 无需数据，与环境互动 |
| **目标** | 预测标签 | 发现结构 | 最大化奖励 |
| **反馈** | 直接（正确答案） | 无反馈 | 延迟（奖励/惩罚） |
| **应用** | 分类、回归 | 聚类、降维 | 游戏、机器人 |
| **难度** | 相对简单 | 中等 | 最难 |
| **前端相关** | ⭐⭐⭐ | ⭐⭐ | ⭐ |

---

## 2.3 常见算法：回归、分类、聚类

掌握这几个最常用的算法。

### 线性回归（Linear Regression）

**用途**：预测数值。

**核心思想**：找到一条直线，最好地拟合数据点。

**直观理解**：
```
数据点：
(面积 50㎡, 价格 200 万)
(面积 80㎡, 价格 320 万)
(面积 100㎡, 价格 400 万)

拟合直线：
价格 = 4 × 面积 + 0

预测：
面积 120㎡ → 价格 = 4 × 120 = 480 万
```

**公式**：
```
y = wx + b

y: 预测值（房价）
x: 输入特征（面积）
w: 权重（每平米价格）
b: 偏置（基础价格）

训练目标：找到最好的 w 和 b
```

**代码示例**：
```javascript
// 简单线性回归实现
class LinearRegression {
  constructor() {
    this.w = 0;  // 权重
    this.b = 0;  // 偏置
  }
  
  // 训练：找到最好的 w 和 b
  fit(X, y) {
    const n = X.length;
    
    // 计算均值
    const xMean = X.reduce((a, b) => a + b, 0) / n;
    const yMean = y.reduce((a, b) => a + b, 0) / n;
    
    // 计算权重 w
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < n; i++) {
      numerator += (X[i] - xMean) * (y[i] - yMean);
      denominator += (X[i] - xMean) ** 2;
    }
    this.w = numerator / denominator;
    
    // 计算偏置 b
    this.b = yMean - this.w * xMean;
  }
  
  // 预测
  predict(x) {
    return this.w * x + this.b;
  }
}

// 使用
const model = new LinearRegression();
model.fit([50, 80, 100, 120], [200, 320, 400, 480]);
console.log(model.predict(150)); // 输出：600（万）
```

### 逻辑回归（Logistic Regression）

**用途**：二分类（是/否）。

**名字误导**：虽然叫"回归"，但实际是分类算法。

**核心思想**：用 Sigmoid 函数把线性回归的输出压缩到 0-1 之间，表示概率。

**Sigmoid 函数**：
```
σ(x) = 1 / (1 + e^(-x))

输入：任意实数
输出：0 到 1 之间

x → -∞: σ(x) → 0
x → +∞: σ(x) → 1
x = 0: σ(x) = 0.5
```

**决策规则**：
```
如果 σ(x) > 0.5 → 预测为 1（是）
如果 σ(x) < 0.5 → 预测为 0（否）
```

**应用场景**：
- 垃圾邮件检测（是/否）
- 用户是否会购买（会/不会）
- 疾病诊断（患病/健康）
- 信用评估（通过/拒绝）

**代码示例**：
```javascript
// Sigmoid 函数
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// 逻辑回归预测
function logisticRegression(features, weights, bias) {
  // 线性组合
  let z = bias;
  for (let i = 0; i < features.length; i++) {
    z += features[i] * weights[i];
  }
  
  // Sigmoid 激活
  const probability = sigmoid(z);
  
  // 分类
  return probability > 0.5 ? 1 : 0;
}

// 使用：垃圾邮件检测
const weights = [0.5, 0.3, -0.2];  // 特征权重
const bias = -0.1;

const email1 = [1, 0, 1];  // 包含"中奖"、不包含"免费"、发件人可疑
const prediction = logisticRegression(email1, weights, bias);
console.log(prediction); // 1（垃圾邮件）
```

### 决策树（Decision Tree）

**用途**：分类或回归。

**核心思想**：通过一系列 if-else 规则做决策。

**直观理解**：
```
判断是否适合户外运动：

开始
  ↓
天气晴朗？
  ├─ 是 → 湿度正常？
  │        ├─ 是 → 适合 ✅
  │        └─ 否 → 不适合 ❌
  └─ 否 → 刮风？
           ├─ 是 → 不适合 ❌
           └─ 否 → 适合 ✅
```

**树的结构**：
```
根节点（天气晴朗？）
  ├── 是 → 内部节点（湿度正常？）
  │         ├── 是 → 叶节点（适合 ✅）
  │         └─ 否 → 叶节点（不适合 ❌）
  └─ 否 → 内部节点（刮风？）
            ├── 是 → 叶节点（不适合 ❌）
            └─ 否 → 叶节点（适合 ✅）
```

**优点**：
- 容易理解和解释
- 不需要特征缩放
- 可以处理数值和类别特征

**缺点**：
- 容易过拟合
- 对数据变化敏感

**代码示例**：
```javascript
// 简化的决策树
class DecisionTree {
  predict(weather, humidity, windy) {
    if (weather === '晴朗') {
      if (humidity === '正常') {
        return '适合';
      } else {
        return '不适合';
      }
    } else {
      if (windy) {
        return '不适合';
      } else {
        return '适合';
      }
    }
  }
}

// 使用
const tree = new DecisionTree();
console.log(tree.predict('晴朗', '正常', false)); // 适合
console.log(tree.predict('阴天', false, false)); // 适合
```

### 随机森林（Random Forest）

**用途**：分类或回归。

**核心思想**：多个决策树投票。

**类比**：
```
一个问题问一个专家，可能出错
问 100 个专家，投票决定，更可靠

随机森林 = 100 棵决策树
```

**工作原理**：
```
1. 从训练数据中有放回地采样（Bootstrap）
2. 每棵树用不同的特征子集训练
3. 预测时，所有树投票
4. 多数票获胜
```

**优点**：
- 比单棵决策树准确
- 不容易过拟合
- 可以处理高维数据

**缺点**：
- 不如单棵树容易解释
- 训练和预测较慢

**代码示例**：
```javascript
// 随机森林预测（概念演示）
class RandomForest {
  constructor(numTrees = 10) {
    this.trees = [];
    this.numTrees = numTrees;
  }
  
  // 训练
  fit(data) {
    for (let i = 0; i < this.numTrees; i++) {
      // 1. Bootstrap 采样
      const sample = bootstrapSample(data);
      
      // 2. 训练决策树
      const tree = trainDecisionTree(sample);
      this.trees.push(tree);
    }
  }
  
  // 预测
  predict(input) {
    // 1. 每棵树预测
    const predictions = this.trees.map(tree => tree.predict(input));
    
    // 2. 投票
    const votes = {};
    for (const pred of predictions) {
      votes[pred] = (votes[pred] || 0) + 1;
    }
    
    // 3. 返回得票最多的
    return Object.keys(votes).reduce((a, b) => 
      votes[a] > votes[b] ? a : b
    );
  }
}
```

### K-Means 聚类

**用途**：无监督聚类。

**核心思想**：把数据分成 K 组，组内相似，组间不同。

**工作原理**：
```
1. 随机选 K 个点作为初始中心
2. 把每个点分配到最近的中心
3. 重新计算每个组的中心
4. 重复 2-3，直到中心不再变化
```

**可视化**：
```
初始：随机 3 个中心（红、绿、蓝）
  ● ● ● ●    ★（红）
  ● ● ● ●
    ★（绿）      ● ● ●
              ★（蓝）

迭代 1：分配点到最近的中心
  ● ● ● ●    ★（红）
  ● ● ● ●
    ★（绿）      ● ● ●
              ★（蓝）

迭代 2：重新计算中心
      ...

收敛：中心不再变化
  红色组 | 绿色组 | 蓝色组
```

**应用场景**：
- 用户分群
- 图像压缩（颜色聚类）
- 异常检测
- 数据探索

---

## 2.4 训练集、验证集、测试集

这是机器学习的基础概念，必须搞懂。

### 为什么要分成三份？

**问题**：如何评估模型好不好？

**错误做法**：用训练数据评估
```
用训练数据评估 → 模型可能死记硬背 → 评估结果虚高
类比：用做过的题考试 → 分数高不代表真会
```

**正确做法**：用没见过的数据评估
```
训练集：学习用（60-80%）
验证集：调参数用（10-20%）
测试集：最终评估用（10-20%）

类比：
训练集 = 练习题（带答案）
验证集 = 模拟考（带答案，用来调整学习方法）
测试集 = 高考（不带答案，最终评估）
```

### 数据划分示例

```javascript
// 假设有 10000 条数据
const allData = loadData(); // 10000 条

// 划分：8:1:1
const shuffled = shuffle(allData);

const trainSize = Math.floor(shuffled.length * 0.8);
const valSize = Math.floor(shuffled.length * 0.1);

const trainData = shuffled.slice(0, trainSize);        // 8000 条
const valData = shuffled.slice(trainSize, trainSize + valSize); // 1000 条
const testData = shuffled.slice(trainSize + valSize);  // 1000 条

// 使用
model.train(trainData);           // 用训练集学习
model.tune(valData);              // 用验证集调参
const finalScore = model.test(testData);  // 用测试集评估
```

### 各部分的作用

**训练集（Training Set）**：
- 用途：训练模型，学习参数
- 大小：60-80%
- 特点：模型会"看到"这些数据

**验证集（Validation Set）**：
- 用途：调整超参数、选择模型
- 大小：10-20%
- 特点：模型间接"看到"（用于调参，不用于学习）
- 使用场景：
  - 选择学习率
  - 选择网络层数
  - 选择正则化强度
  - 早停（Early Stopping）

**测试集（Test Set）**：
- 用途：最终评估模型性能
- 大小：10-20%
- 特点：模型完全没"看到"过
- 重要：只能用一次！用完就不能再调模型了

### 交叉验证（Cross-Validation）

**问题**：数据太少，不够分三份怎么办？

**解决方案**：交叉验证

**K 折交叉验证**：
```
数据分成 K 份（通常 K=5 或 10）

第 1 轮：用第 1 份做验证，其他做训练
第 2 轮：用第 2 份做验证，其他做训练
...
第 K 轮：用第 K 份做验证，其他做训练

最终得分 = K 轮的平均分
```

**可视化**（5 折）：
```
数据：[1][2][3][4][5]

轮次 1：[训练][训练][训练][训练][验证]
轮次 2：[训练][训练][训练][验证][训练]
轮次 3：[训练][训练][验证][训练][训练]
轮次 4：[训练][验证][训练][训练][训练]
轮次 5：[验证][训练][训练][训练][训练]

平均分 = (得分 1 + 得分 2 + 得分 3 + 得分 4 + 得分 5) / 5
```

**代码示例**：
```javascript
// K 折交叉验证
function crossValidate(model, data, k = 5) {
  const foldSize = Math.floor(data.length / k);
  const scores = [];
  
  for (let i = 0; i < k; i++) {
    // 划分
    const valStart = i * foldSize;
    const valEnd = valStart + foldSize;
    
    const valData = data.slice(valStart, valEnd);
    const trainData = [
      ...data.slice(0, valStart),
      ...data.slice(valEnd)
    ];
    
    // 训练和评估
    model.train(trainData);
    const score = model.evaluate(valData);
    scores.push(score);
  }
  
  // 返回平均分
  const avgScore = scores.reduce((a, b) => a + b, 0) / k;
  console.log(`交叉验证得分：${avgScore} (${scores.join(', ')})`);
  
  return avgScore;
}
```

---

## 2.5 过拟合、欠拟合、正则化

这是面试必考题，也是实际训练中最常遇到的问题。

### 欠拟合（Underfitting）

**定义**：模型太简单，连训练数据都学不好。

**表现**：
- 训练集准确率低
- 测试集准确率低
- 模型没学会规律

**类比**：
```
学生太懒，练习题都不做
结果：平时作业差，考试也差
```

**可视化**：
```
数据点：● ● ● ● ●

欠拟合模型：一条直线
───────

明显拟合不好
```

**原因**：
1. 模型太简单（层数太少、参数太少）
2. 特征不够（信息不足）
3. 训练时间不够

**解决方法**：
1. 增加模型复杂度（更多层、更多参数）
2. 增加特征
3. 训练更久
4. 减少正则化

### 过拟合（Overfitting）

**定义**：模型太复杂，把训练数据背下来了，但没学会规律。

**表现**：
- 训练集准确率很高（99%+）
- 测试集准确率低
- 泛化能力差

**类比**：
```
学生死记硬背练习题答案
结果：平时作业全对，考试换题就不会了
```

**可视化**：
```
数据点：●   ●   ●   ●   ●

过拟合模型：一条弯弯曲曲的线，穿过每个点
╭─╮   ╭─╮
│ │   │ │
╰─╯   ╰─╯

训练集完美拟合，但新数据会很差
```

**原因**：
1. 模型太复杂（层数太多、参数太多）
2. 训练数据太少
3. 训练时间太长
4. 没有正则化

**解决方法**：
1. 增加训练数据
2. 减少模型复杂度
3. 正则化（L1、L2、Dropout）
4. 早停（Early Stopping）
5. 数据增强（Data Augmentation）

### 正则化（Regularization）

**定义**：防止过拟合的技术。

**核心思想**：限制模型复杂度，让它学一般规律，而不是死记硬背。

#### L1 正则化（Lasso）

**公式**：
```
损失函数 += λ × Σ|w|

λ: 正则化强度
w: 权重
```

**效果**：
- 让一些权重变成 0
- 相当于特征选择
- 模型更稀疏

**类比**：
```
限制学生只能记住最重要的知识点
次要的知识点可以忽略
```

#### L2 正则化（Ridge）

**公式**：
```
损失函数 += λ × Σw²
```

**效果**：
- 让权重变小，但不为 0
- 所有特征都用，但影响小
- 最常用

**类比**：
```
限制学生每个知识点都不能花太多时间
要均衡发展
```

#### Dropout

**定义**：训练时随机"关掉"一些神经元。

**工作原理**：
```
训练时：
输入 → [神经元 1][神经元 2][神经元 3][神经元 4] → 输出
              ↓           ↓
           （关掉）    （关掉）

实际生效：
输入 → [神经元 1][        ][        ][神经元 4] → 输出
```

**效果**：
- 防止神经元之间过度依赖
- 相当于训练多个子模型
- 测试时所有神经元都用

**代码示例**：
```javascript
// TensorFlow.js 中使用 Dropout
const model = tf.sequential();
model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [10] }));
model.add(tf.layers.dropout({ rate: 0.5 }));  // 50% Dropout
model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
model.add(tf.layers.dropout({ rate: 0.5 }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
```

#### 早停（Early Stopping）

**定义**：训练过程中监控验证集，性能不再提升就停止。

**工作原理**：
```
训练轮次    训练准确率    验证准确率
1          60%          58%
2          70%          68%
3          80%          75%
4          90%          78%   ← 最佳点
5          95%          77%   ← 开始下降（过拟合）
6          98%          76%
7          99%          75%

在第 4 轮停止，保存模型
```

**代码示例**：
```javascript
// TensorFlow.js 早停回调
const model = await tf.sequential();
// ... 定义模型

await model.fit(trainData, trainLabels, {
  epochs: 100,
  validationData: [valData, valLabels],
  callbacks: [
    tf.callbacks.earlyStopping({
      monitor: 'val_loss',      // 监控验证集损失
      patience: 5,              // 5 轮不提升就停止
      restoreBestWeights: true, // 恢复最佳权重
    })
  ]
});
```

### 过拟合 vs 欠拟合对比

| 对比项 | 欠拟合 | 正常 | 过拟合 |
|--------|--------|------|--------|
| **训练准确率** | 低 | 高 | 很高 |
| **测试准确率** | 低 | 高 | 低 |
| **模型复杂度** | 太低 | 适中 | 太高 |
| **偏差 - 方差** | 高偏差 | 平衡 | 高方差 |
| **解决** | 增加复杂度 | 保持 | 正则化 |

### 诊断流程

```
1. 计算训练集和测试集准确率

2. 如果训练集准确率低 → 欠拟合
   解决：增加模型复杂度、增加特征、训练更久

3. 如果训练集准确率高，测试集准确率低 → 过拟合
   解决：增加数据、正则化、减少复杂度、早停

4. 如果都高 → 完美！
```

---

## 2.6 损失函数、梯度下降

这是模型训练的核心，面试必考。

### 损失函数（Loss Function）

**定义**：衡量模型预测和真实答案的差距。

**目标**：最小化损失函数。

**类比**：
```
损失函数 = 考试错题数

目标：错题越少越好（损失越小越好）
```

### 常见损失函数

#### 均方误差（MSE）- 回归问题

**公式**：
```
MSE = (1/n) × Σ(预测值 - 真实值)²
```

**解释**：
- 预测和真实的差的平方
- 平方是为了让正负误差不抵消
- 除以 n 是求平均

**代码示例**：
```javascript
function mse(predictions, labels) {
  const n = predictions.length;
  let sum = 0;
  
  for (let i = 0; i < n; i++) {
    const diff = predictions[i] - labels[i];
    sum += diff * diff;  // 平方
  }
  
  return sum / n;
}

// 使用
const predictions = [2.1, 3.9, 5.2];
const labels = [2.0, 4.0, 5.0];
console.log(mse(predictions, labels)); // 0.02
```

**特点**：
- 对大误差敏感（平方放大）
- 连续可导
- 回归问题最常用

#### 交叉熵损失（Cross-Entropy）- 分类问题

**公式**（二分类）：
```
Loss = -[y × log(p) + (1-y) × log(1-p)]

y: 真实标签（0 或 1）
p: 预测概率
```

**解释**：
- y=1 时：Loss = -log(p)，预测概率越高损失越小
- y=0 时：Loss = -log(1-p)，预测概率越低损失越小

**代码示例**：
```javascript
function binaryCrossEntropy(predictions, labels) {
  const n = predictions.length;
  let loss = 0;
  
  for (let i = 0; i < n; i++) {
    const p = predictions[i];
    const y = labels[i];
    
    // 防止 log(0)
    const epsilon = 1e-7;
    const pClipped = Math.max(epsilon, Math.min(1 - epsilon, p));
    
    loss += -(y * Math.log(pClipped) + (1 - y) * Math.log(1 - pClipped));
  }
  
  return loss / n;
}

// 使用
const predictions = [0.9, 0.1, 0.8];  // 预测概率
const labels = [1, 0, 1];              // 真实标签
console.log(binaryCrossEntropy(predictions, labels)); // 0.14
```

**特点**：
- 适合概率输出
- 对错误预测惩罚大
- 分类问题最常用

### 梯度下降（Gradient Descent）

**定义**：通过迭代找到损失函数最小值的算法。

**核心思想**：往梯度（导数）的反方向走，函数值会下降。

**类比**：
```
你在山上，想走到最低点（山谷）

方法：
1. 看脚下哪个方向最陡（计算梯度）
2. 往那个方向走一步（更新参数）
3. 重复 1-2，直到到谷底
```

**可视化**：
```
损失函数曲线：
  Loss
   ↑
   │    ╱
   │   ╱
   │  ╱
   │ ╱
   │╱___________→ 参数 w
   
从高处往低处走，找到最低点
```

### 梯度下降公式

```
w_new = w_old - learning_rate × gradient

w: 参数（权重）
learning_rate: 学习率（步长）
gradient: 梯度（导数，∂Loss/∂w）
```

**解释**：
- gradient > 0：函数在上升，往左走（减）
- gradient < 0：函数在下降，往右走（加）
- learning_rate：走多大步

### 学习率的选择

**太大**：
```
学习率 = 1.0

迭代 1：w = 5
迭代 2：w = -3  （跨过头了）
迭代 3：w = 7   （又跨过头）
迭代 4：w = -1
...

结果：在最低点附近震荡，不收敛
```

**太小**：
```
学习率 = 0.001

迭代 1：w = 5
迭代 2：w = 4.999
迭代 3：w = 4.998
...
迭代 1000：w = 4

结果：走得慢，要很久才到
```

**合适**：
```
学习率 = 0.1

迭代 1：w = 5
迭代 2：w = 4
迭代 3：w = 3
迭代 4：w = 2
迭代 5：w = 1  （到达最低点）

结果：刚好收敛
```

**常用学习率**：
```
0.1, 0.01, 0.001, 0.0001

建议：从 0.01 开始，不行再调
```

### 梯度下降变体

#### Batch Gradient Descent（批量梯度下降）

**方法**：用所有训练数据计算梯度。

**优点**：
- 稳定，收敛平滑

**缺点**：
- 数据多时慢
- 内存占用大

#### SGD（随机梯度下降）

**方法**：每次用一个样本计算梯度。

**优点**：
- 快
- 可以在线学习

**缺点**：
- 波动大
- 可能不收敛到最优

#### Mini-batch Gradient Descent（小批量梯度下降）

**方法**：每次用一小批数据（如 32、64、128 个）。

**优点**：
- 平衡速度和稳定性
- 最常用

**代码示例**：
```javascript
// 简化的梯度下降实现
class GradientDescent {
  constructor(learningRate = 0.01) {
    this.lr = learningRate;
    this.w = 0;
    this.b = 0;
  }
  
  // 计算梯度
  computeGradient(X, y, predictions) {
    const n = X.length;
    
    // 对 w 的梯度
    let dw = 0;
    for (let i = 0; i < n; i++) {
      dw += (predictions[i] - y[i]) * X[i];
    }
    dw /= n;
    
    // 对 b 的梯度
    let db = 0;
    for (let i = 0; i < n; i++) {
      db += predictions[i] - y[i];
    }
    db /= n;
    
    return { dw, db };
  }
  
  // 更新参数
  update(dw, db) {
    this.w -= this.lr * dw;
    this.b -= this.lr * db;
  }
  
  // 训练
  train(X, y, epochs = 1000) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      // 1. 预测
      const predictions = X.map(x => this.w * x + this.b);
      
      // 2. 计算梯度
      const { dw, db } = this.computeGradient(X, y, predictions);
      
      // 3. 更新参数
      this.update(dw, db);
      
      // 打印进度
      if (epoch % 100 === 0) {
        const loss = this.computeLoss(predictions, y);
        console.log(`Epoch ${epoch}: Loss = ${loss}`);
      }
    }
  }
  
  computeLoss(predictions, y) {
    let loss = 0;
    for (let i = 0; i < predictions.length; i++) {
      const diff = predictions[i] - y[i];
      loss += diff * diff;
    }
    return loss / predictions.length;
  }
}

// 使用
const model = new GradientDescent(0.01);
model.train([1, 2, 3, 4], [2, 4, 6, 8], 1000);
console.log(`w = ${model.w}, b = ${model.b}`);
// 输出：w ≈ 2, b ≈ 0
```

---

## 2.7 前端需要知道的 ML 知识边界

作为前端开发，不需要掌握所有 ML 知识。

### 必须掌握的

```
✅ 核心概念
- 机器学习 vs 传统编程的区别
- 监督学习、无监督学习、强化学习
- 训练集、验证集、测试集
- 过拟合、欠拟合、正则化
- 损失函数、梯度下降（理解概念）

✅ 常见算法
- 线性回归、逻辑回归
- 决策树、随机森林
- K-Means 聚类

✅ 评估指标
- 准确率、精确率、召回率、F1
- MSE、交叉熵

✅ 实际应用
- 如何调用 ML 模型（API）
- 如何集成到前端应用
```

### 了解就行的

```
⭕ 算法推导
- 反向传播公式（不用手推）
- SVM 的拉格朗日对偶（不用懂）
- PCA 的特征值分解（不用算）

⭕ 底层实现
- 卷积的具体计算（不用手算）
- 矩阵求导（不用会）
- 优化算法的数学证明（不用看）

⭕ 训练技巧
- 超参数调优（用 AutoML）
- 模型架构设计（用现成的）
```

### 完全不用管的

```
❌ 数学证明
- 收敛性证明
- 泛化界证明
- 各种不等式推导

❌ 底层优化
- CUDA 编程
- 分布式训练
- 模型量化、剪枝

❌ 研究前沿
- 最新论文
- SOTA 模型
- 理论突破
```

### 前端的优势

```
作为前端，你在 ML 应用中的优势：

1. 用户体验
   - 设计友好的 ML 应用界面
   - 处理加载状态、错误提示
   - 流式输出、打字机效果

2. 工程化
   - API 集成
   - 状态管理
   - 性能优化

3. 全栈能力
   - 前端 + 后端 + ML API
   - 完整的产品思维
```

### 学习建议

```
1. 先理解概念（这章的内容）
2. 学会调用 API（第 7 章）
3. 做项目实践（RAG、Agent）
4. 遇到不懂的再查

不要：
1. 一开始就啃数学书
2. 试图推导所有公式
3. 追求完全理解再动手
```

---

## 本章小结

### 核心概念

```
机器学习 vs 传统编程：从数据学习 vs 人工写规则
监督学习：有标签（分类、回归）
无监督学习：无标签（聚类、降维）
强化学习：试错学习（奖励/惩罚）

训练集/验证集/测试集：学习/调参/评估
过拟合/欠拟合：太复杂/太简单
正则化：防止过拟合（L1、L2、Dropout、早停）

损失函数：衡量误差（MSE、交叉熵）
梯度下降：优化算法（往梯度反方向走）
```

### 面试必背

```
1. 机器学习三种类型的区别和例子
2. 过拟合的表现和解决方法
3. 训练集、验证集、测试集的作用
4. 梯度下降的原理
5. 常见损失函数及其适用场景
```

### 下一步

下一章：[深度学习入门](./03-deep-learning-basics.md)

---

**💡 学习建议**：
1. 把每个概念用自己的话讲一遍
2. 跑一下代码示例（TensorFlow.js）
3. 做课后自测题
4. 不理解的地方先标记，后面会反复遇到

# 09 术语表与延伸阅读

## 术语表

- `Token`
  模型处理的基本离散单位，不等于单词或汉字。
- `Tokenizer`
  把原始文本转换成 token 序列和 ID 的组件。
- `Vocabulary`
  token 到整数 ID 的映射表。
- `Embedding`
  把离散 token ID 映射到连续向量空间的矩阵。
- `Hidden State`
  模型中间层对当前上下文的表示。
- `Attention`
  根据相关性从上下文中加权读取信息的机制。
- `Self-Attention`
  序列内部各位置彼此做 attention。
- `Q/K/V`
  Query、Key、Value 三种投影，用于计算注意力。
- `Causal Mask`
  限制当前位置不能看到未来 token 的掩码。
- `Logits`
  softmax 前的原始分数向量。
- `Decoding`
  从概率分布中选出实际输出 token 的过程。
- `Prefill`
  处理已有输入并构建初始 KV Cache 的阶段。
- `Decode`
  基于缓存逐 token 生成后续输出的阶段。
- `KV Cache`
  历史 token 的 Key/Value 缓存，用于避免重复计算。
- `Quantization`
  用更低比特表示数值的技术。
- `Weight Quantization`
  压缩模型权重。
- `KV Quantization`
  压缩运行时缓存中的 K/V。
- `SFT`
  Supervised Fine-Tuning，用标注样本继续训练使模型更会按指令做事。
- `RLHF`
  Reinforcement Learning from Human Feedback，用人类偏好优化模型输出。
- `DPO`
  Direct Preference Optimization，一种更简洁的偏好优化方法。
- `Reasoning Model`
  更擅长多步规划、验证和复杂任务完成的模型。
- `GGUF`
  本地模型部署里常见的权重文件格式。
- `llama.cpp`
  轻量本地大模型推理框架。
- `Gradient`
  损失函数对参数变化的敏感方向，是训练更新参数的核心信号。
- `Optimizer`
  根据梯度决定参数怎么更新的算法，例如 Adam、AdamW。
- `Learning Rate`
  每一步参数更新走多远的控制量。
- `Embedding Model`
  把文本或对象编码为向量的模型，常用于检索、聚类和召回。
- `Vector Search`
  在向量空间中查找最相近对象的搜索方式。
- `RAG`
  Retrieval-Augmented Generation，把外部检索结果注入上下文。
- `Chunking`
  在 RAG 中把文档切分为更适合索引和召回的小片段。
- `Rerank`
  对召回候选再次排序，以提高最终相关性。
- `LoRA`
  一种参数高效微调方法，通过低秩增量适配模型。
- `QLoRA`
  在量化基础模型上训练 LoRA adapter 的方法。
- `Distillation`
  让较小 student 模型学习较强 teacher 模型能力的技术。
- `Evaluation Set`
  用于比较模型或系统改动效果的固定测试样本集合。
- `Hallucination`
  模型生成看似合理但缺乏依据或与事实不符的内容。
- `Guardrails`
  用于约束 AI 系统行为的规则、校验器和策略。
- `Agent`
  能够规划、调用工具、读取结果并推进任务的 AI 系统。
- `Function Calling`
  模型以结构化方式表达工具调用意图的接口协议。

## 怎么使用这份术语表

这份术语表不是为了背定义，而是为了建立“术语之间的依赖关系”。建议按下面的顺序去理解：

1. 先掌握 `Token -> Embedding -> Hidden State`
2. 再掌握 `Q/K/V -> Attention -> Causal Mask`
3. 接着掌握 `Prefill -> Decode -> KV Cache`
4. 然后掌握 `Gradient -> Optimizer -> Learning Rate`
5. 最后掌握 `SFT -> RLHF/DPO -> RAG -> Agent`

如果你能把这五条链在脑中连起来，大模型的大多数文章就都能读懂个八九不离十。

## 建议阅读顺序

1. 先读 `00` 到 `04`，把数学、输入、表示、架构打牢。
2. 再读 `05`、`10`、`12`，理解“能力怎么形成”和“训练如何把能力塑形”。
3. 再读 `06`、`07`、`08`，进入推理与部署视角。
4. 最后读 `11`、`13`、`14`，进入真正的应用系统设计。

## 推荐论文与资料

- Vaswani et al., *Attention Is All You Need*
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*
- Kudo and Richardson, *SentencePiece*
- Brown et al., *Language Models are Few-Shot Learners*
- Ouyang et al., *Training language models to follow instructions with human feedback*
- Google Research, *TurboQuant: Redefining AI efficiency with extreme compression*
- Hu et al., *LoRA*
- Lewis et al., *RAG*

## 推荐阅读路径

如果你想按“教材化”的节奏继续深入，建议这样走：

1. 先把 `00` 到 `04` 读到能自己解释公式和数据流。
2. 再把 `05`、`10`、`12` 读到能解释“能力是怎么训练和塑形出来的”。
3. 再把 `06`、`07`、`08` 读到能判断一个推理优化方案到底是在省算力、带宽还是缓存。
4. 最后把 `11`、`13`、`14` 读到能搭一个真正可上线、可评估的 AI 系统。

对开发者来说，这条顺序比一开始就追热点论文更有效。

## 连贯学习建议

如果你觉得“每章都懂一点，但连不起来”，建议把下面两篇一起穿插阅读：

- [15-knowledge-map-and-study-roadmap.md](./15-knowledge-map-and-study-roadmap.md)
- [16-end-to-end-practice-building-an-ai-assistant.md](./16-end-to-end-practice-building-an-ai-assistant.md)

第一篇帮你建立知识地图，第二篇帮你把知识放回真实项目。

## 面向工程实践的下一步

如果你准备把这套知识转成实践，建议继续研究：

- `llama.cpp` 或 `vLLM` 的推理栈
- GGUF / safetensors / tokenizer 文件结构
- FlashAttention / paged attention / speculative decoding
- RAG 系统设计
- Agent 工作流编排
- 微调、LoRA 和蒸馏训练栈
- 评测平台、trace 和在线指标看板

## 一句收束

大模型不是单一技术，而是一条层层叠加的技术链：分词把文本变成离散符号，神经网络把离散符号映射到连续空间，Transformer 负责在上下文里做动态信息读取，预训练和对齐塑造通用能力，推理系统负责把这些能力在现实硬件上高效落地，TurboQuant 则代表了这条链在推理优化方向上的最新推进。

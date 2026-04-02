# 01 分词与 Tokenization

## 这章怎么读

如果你是零基础，先不要纠结 `BPE`、`WordPiece`、`SentencePiece` 这些名字。  
先抓住一件事：模型根本不直接吃“句子”，它先把文本切成 token，再把 token 变成 ID。

读这章时，建议优先回答 3 个问题：

- 文本为什么不能直接喂给模型
- token 为什么不等于“词”或“字”
- 分词方式为什么会影响上下文长度、训练效率和产品成本

## 先记住这条链路

```mermaid
flowchart LR
  A["原始文本"] --> B["Tokenizer"]
  B --> C["Token 序列"]
  C --> D["Token IDs"]
  D --> E["Embedding"]
  E --> F["模型开始计算"]
```

分词是大模型系统的入口。模型本身不理解“字词句”的自然语言形式，它只接受数字张量，因此任何文本输入都必须先转换成一串 token ID。

## 1. 为什么需要分词

文本在计算机里本质上是字符序列，但直接按字符建模会遇到三个问题：

- 序列太长。英语中一个单词往往会被拆成多个字符，中文虽然单字短，但语义单位又经常不是单字。
- 语义太稀疏。字符层面的模式不容易直接对应词义、句法和实体。
- 词表不可控。按“完整单词”建模会遇到大量未登录词，例如新产品名、拼写错误、代码符号和多语言混写。

因此，现代 LLM 采用折中方案：`subword tokenization`。也就是把文本切成“比字符大、比完整单词小”的子词单元。

## 2. 从“词”到“token”

要区分几个相近但不同的概念：

- `character`：字符，例如 `a`、`中`、`_`
- `word`：语言学上的词，中文里边界并不天然明确
- `token`：分词器输出的基本单元，是模型真正处理的单位
- `token id`：token 在词表中的整数编号
- `vocabulary`：词表，记录“token <-> id”的映射

大模型里的 `token` 并不等于“单词”或“汉字”。例如：

- `ChatGPT` 可能被切成多个 token
- `transformers` 可能被切成 `transform` 和 `ers`
- 一个中文词组有时会是一个 token，有时会拆成几个字

## 3. 典型的 tokenizer 流程

```mermaid
flowchart LR
  A["原始文本"] --> B["Normalization\n大小写、空格、Unicode 规范化"]
  B --> C["Pre-tokenization\n按空格/标点/字节做初步切分"]
  C --> D["Subword Algorithm\nBPE / WordPiece / Unigram"]
  D --> E["Post-processing\n加 BOS/EOS/特殊 token"]
  E --> F["Token IDs"]
```

### 3.1 Normalization

这一步用于把文本变成更稳定的输入形式，例如：

- 全角半角统一
- Unicode 规范化
- 空白字符折叠
- 某些模型会做大小写归一

这一步的目标是减少“表面不同、实质相同”的写法造成的词表碎片化。

### 3.2 Pre-tokenization

先做一层粗切分，例如按空格、标点、数字边界、字节边界切开。这样能降低后续子词搜索的复杂度。

### 3.3 子词算法

这是 tokenizer 的核心。主流方法有三类。

## 4. 主流分词算法

### 4.1 BPE

`Byte Pair Encoding` 的直觉是：从细粒度单位开始，反复合并最常同时出现的相邻片段。

训练过程可以理解为：

1. 把文本拆成初始单元，例如字符或字节。
2. 统计相邻单元对的出现频率。
3. 每次把最高频的 pair 合并成一个新 token。
4. 重复很多轮，直到达到目标词表大小。

优点：

- 简单、高效、实现广泛
- 高频片段会自然形成稳定 token
- 对开放词汇比较友好

缺点：

- 纯频率驱动，不直接建模“整词概率”
- 某些语言和场景下会产生不够自然的切分

### 4.2 WordPiece

WordPiece 和 BPE 很像，但它不是单纯看 pair 频次，而是更偏向“合并后是否提高语言建模效果”。BERT 系列广泛使用它。

优点：

- 在语言模型训练中表现稳定
- 对词片边界控制相对更好

### 4.3 Unigram / SentencePiece

Unigram 的思路与“不断合并”不同。它先准备一个较大的候选词表，然后反复删掉价值较低的 token，使剩余词表对训练语料的整体概率最好。

SentencePiece 的重要意义在于：

- 它可以直接在原始文本上训练，不依赖预先按空格切词
- 对中文、日文等“词边界不明显”的语言非常实用
- 很多现代 LLM 都采用它或其变体

## 5. 为什么中文分词更特别

英语天然有空格边界，中文没有，因此中文 tokenizer 会更依赖子词统计，而不是先验词边界。

例如一句中文：

`今天天气不错，我们去看电影吧。`

可能的切分方式有很多种：

- 按字：`今 / 天 / 天 / 气 / 不 / 错`
- 按词：`今天 / 天气 / 不错`
- 按子词混合：`今天 / 天 / 气 / 不错`

不同 tokenizer 的设计会直接影响：

- 序列长度
- 训练稳定性
- 推理速度
- 多语言混合的处理效果

## 6. Tokenization 为什么会影响模型能力

分词并不只是预处理细节，它会深刻影响模型的上限和成本。

### 6.1 上下文长度

上下文窗口通常按 token 计数，不按字符计数。分词越碎，单位文本消耗的 token 越多，可容纳的内容越少。

### 6.2 训练效率

序列更长意味着：

- attention 计算更多
- 显存占用更高
- 批大小更小

### 6.3 表达能力

如果分词过粗，模型会失去拆解罕见词的能力；如果过细，模型要花很多层才能重新组合出高级语义。

### 6.4 产品成本

很多 API 按 token 计费，本地部署则按 token 消耗计算和内存。因此 tokenizer 实际上会影响商业成本。

## 7. 特殊 token 的作用

除了普通文本 token，模型还会定义一些特殊 token：

- `BOS`：序列开始
- `EOS`：序列结束
- `PAD`：对齐批次长度
- `UNK`：未知 token
- `SYS` / `USER` / `ASSISTANT`：聊天模板角色标记
- 工具调用、函数参数、代码块边界等控制符号

聊天模型并不是简单把你的消息原样送进去，而是会套一层 `chat template`，把角色、系统指令、工具定义等都编码成 token 序列。

## 8. Byte-level tokenizer 为什么流行

GPT-2 之后，字节级 tokenizer 很流行。它不依赖“合法字符集”假设，而是让任何输入都能映射到字节序列。

好处是：

- 几乎没有真正意义上的 OOV
- 对代码、emoji、控制符、多语言混写更稳
- 工程上更鲁棒

代价是：

- 某些输入会被切得更碎
- 可读性和词片直觉可能变差

## 9. 开发者应关注什么

- 不同模型的 tokenizer 不能随便混用
- token 数量要用模型自带 tokenizer 估算
- 提示词工程的“长短感受”必须以 token 为准
- 代码、JSON、表格通常比自然语言更耗 token
- 微调、蒸馏、RAG 切片都要考虑 tokenizer 行为

## 10. 从优化目标看，分词器到底在逼近什么

一个真正工程化的 tokenizer 设计，本质上是在解一个受约束的压缩问题：

- 输入空间是开放的自然语言和代码
- 词表大小必须有限
- 序列长度必须尽量短
- 同时还要保留足够强的可组合性

如果形式化一点，可以把目标理解为：

在固定词表预算 `|V| <= K` 下，找到一种切分规则，使得训练语料被编码后的总成本最小，同时让下游语言模型更容易学习。

“总成本”通常不是单一指标，而是多个目标的折中：

- 语料平均 token 长度尽量短
- 词表不要过大，否则 embedding 和 softmax 代价上升
- 长尾词要可拆解，否则 OOV 或泛化变差
- 切分边界要尽量稳定，否则模型难以形成一致语义

这也是为什么 tokenizer 设计没有绝对最优解，只有面向具体语料和模型规模的工程折中。

## 11. BPE 的训练过程：为什么它像一种贪心压缩

用一个小例子看 BPE 会更直观。假设语料里经常出现：

- `low`
- `lower`
- `lowest`

初始时按字符切分：

- `l o w`
- `l o w e r`
- `l o w e s t`

如果 `l + o` 高频共现，先合成 `lo`；之后 `lo + w` 又高频，就变成 `low`；再往后可能得到 `low + er`、`low + est`。

这和压缩算法的直觉很像：不断把最常同时出现的局部模式，提升为一个新的原子单位。

一个简化版伪代码如下：

```text
initialize vocabulary with characters or bytes
repeat until vocab size reaches limit:
  count adjacent pair frequencies on corpus
  select the most frequent pair (a, b)
  merge (a, b) into a new token ab
  replace all occurrences in corpus
```

它为什么有效：

- 高频模式会被自然“打包”，降低序列长度
- 低频词仍能退回更细粒度单元，不至于完全无法表示

它为什么不完美：

- 这是局部贪心，不保证全局最优
- 频率最高不等于语义最合理
- 训练语料一旦变化很大，旧词表的最优性会下降

## 12. Unigram / SentencePiece 为什么常被认为更“语言学友好”

Unigram 不是从小单元往上合并，而是从一个较大的候选词表开始，反复删除价值低的 token。它背后的直觉更接近概率建模：

- 一种切分方式如果能让整段文本概率更高，就更合理
- token 的价值由它对整体语料似然的贡献决定

这带来两个实际好处：

- 更容易得到多种可接受切分，而不是被单一路径锁死
- 对中日韩和多语言混合文本往往更稳

这也是为什么很多现代基础模型更偏向 SentencePiece 风格方案。

## 13. Tokenization 对训练图的影响，比想象中更直接

很多人把 tokenizer 当成“离线预处理”，但它其实会直接改变训练图的多个关键张量尺寸：

- 序列长度 `n`
- 词表大小 `V`
- embedding 参数量
- output head 的类别数

例如，固定一段文本：

- 如果被编码成 1,200 个 token，attention 复杂度与长度相关
- 如果换一个 tokenizer 变成 1,600 个 token，训练和推理成本会显著上升

在自注意力模型里，序列长度增加带来的代价通常非常敏感，因此 tokenizer 的“碎片化程度”会直接影响训练预算。

## 14. 代码和结构化文本为什么特别考验 tokenizer

大模型在代码、JSON、日志和配置文件上的表现，往往和 tokenizer 有强耦合。

原因包括：

- 标点、缩进、括号、分隔符密度高
- 标识符存在大量驼峰、下划线、数字拼接
- 一个小小的切分偏差就会让语法单元被拆散

例如变量名：

`user_profile_v2_cache_key`

如果 tokenizer 能把它较合理地拆成：

- `user`
- `_profile`
- `_v2`
- `_cache`
- `_key`

模型就更容易学到跨项目复用的模式；如果被切成非常零碎的字节片段，模型虽然仍能学，但学习效率会下降。

## 15. Tokenizer mismatch 是实战中的常见坑

实际开发里经常出现以下问题：

- 用错 tokenizer 估算 token 数，导致上下文截断
- 训练和推理阶段 tokenizer 不一致
- 模型升级后 tokenizer 改了，缓存或切片逻辑却没同步更新
- RAG chunk 是按字符切的，而不是按 token 切的

这些问题的后果不是“有点不准”，而可能是：

- 输入被截断在错误位置
- 检索片段和生成预算不匹配
- 工具调用模板被破坏
- 微调样本中的标签边界错位

## 16. 一个更准确的心智模型

可以把 tokenizer 想成编译器前端，但要再加两层理解：

- 它不仅在“词法分析”，也在做压缩
- 它不仅决定输入怎么切，还决定模型的成本曲线

所以 tokenizer 不是中性的格式转换器，而是模型系统设计的一部分。

## 17. 小结

Tokenization 解决的是“如何把开放世界文本转成有限词表中的离散单位”。真正重要的不是它把文本切开了，而是它在词表规模、序列长度、表达能力和工程成本之间做了一次全局折中。

## 参考阅读

- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*
- Kudo and Richardson, *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*
- Karpathy, *Let's build the GPT tokenizer*

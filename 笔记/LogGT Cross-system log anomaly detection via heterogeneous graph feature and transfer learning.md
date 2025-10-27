# LogGT: ==Cross-system== log anomaly detection via heterogeneous graph feature and transfer learning

LogGT：基于异构图特征和迁移学习的跨系统日志异常检测

## 核心问题

- **信息利用不完整**：忽视日志中的组件（如 “Xinetd”“Syslog”）、时间间隔等关键信息，易遗漏执行顺序异常、性能异常（如网络延迟导致的时间间隔变化）。
- **跨系统适配难**：单系统训练模型因日志语法差异（如 Linux 与 Windows 日志表述不同）、目标系统缺乏标注数据，难以迁移应用。
- **语义提取不足**：传统词嵌入（如 Word2vec）受语法格式影响大，无法有效捕捉跨系统日志语义相似性。

## 贡献

• 我们提出了一种异构图建模方法，以有效组合不同组件和日志事件，并设计了GTAT来进一步考虑时间间隔，从而捕捉异构图的高阶复杂信息。

• 我们采用了一种高效的日志语义表示方法，该方法将日志语句作为BERT的输入，并使用归一化流进行优化，从而得到图节点嵌入。这避免了不同系统之间因语法差异而导致的语义混淆。

• 我们采用了一种新颖的领域自适应迁移学习技术，并设计了一种语义加权方法，该方法计算语义权重，并将其与最大均值差异损失相结合，以有效迁移不同软件系统中的异构图信息。

## 前人的方法

![img](https://cdn.xljsci.com/literature/181810556/page3/jg210m.png)

> 日志异常检测方法的比较：Sequence (S)、Graph (G)、log Events (E)、Time (T)、Component (C)、Parameter Value (P)、Level (L)、Adapter (A)、Event Count Vector （EV）、Event Index （EI）、FASTText （FT）、TF- idf （TI）、Word2vec （WV）、Post-processing Algorithms （PA）、Inverse Document Frequency （IF）、Sentence-Bert （SB）、Position Embedding （PE）、Normalizing Flow （NF）、Doc2vec （DV）、Random Forests （RF）、Transformer （TF）、Bi-LSTM-Attention （BA）、LSTM-Attention （LA）、转移概率（TP）、迁移学习（TL）、语义权重（SW）。平滑逆频率（SIF），对抗域自适应（ADA）。

## 设计

### 框架

![img](https://cdn.xljsci.com/literature/181810556/page4/sc5tl4.png)

1.**日志解析与异构图构建**：

- 用 Drain 算法将非结构化日志解析为**结构化模板（日志事件），提取组件、时间戳信息。**
- 构建异构图`G=(P,E,A,R)`：**节点`P`为日志事件，边`E`表示事件执行顺序及频次权重，节点类型`A`对应组件，边类型`R`区分 “同组件”“跨组件” 事件交互，无需手动定义元路径。**

![img](https://cdn.xljsci.com/literature/181810556/page5/ayrym8.png)

![img](https://cdn.xljsci.com/literature/181810556/page9/qbaeqx.png)

![img](https://oss.xljsci.com//literature/181810556/page0/1760148233935.png)

2.**节点嵌入优化**：

- 用 BERT 模型输入完整日志句子（而非单个单词），生成语义向量，避免语法差异导致的语义混淆。
- 用 Normalizing Flow 将向量转化为平滑高斯分布，优化节点嵌入，提升跨系统语义一致性。

> 日志异常检测中，节点嵌入需满足 “可靠性” 与 “完整性” 两大核心需求：前者要求相似语义的日志事件（如 Linux 的 “execute <*>: No such file or directory” 与 Windows 的 “The system cannot find the file specified”）生成高相似度向量，后者要求向量完整保留日志句子的语序、关键信息权重（如 “block” 比 “an” 对语义更重要）。此前方法（如 Word2vec、GloVe）直接基于单个单词或模板索引生成嵌入，易受跨系统日志语法格式差异影响（如不同系统日志的字段顺序、表述风格不同），导致语义混淆 —— 例如相同语义的日志因语法不同生成低相似度向量，或不同语义的日志因包含相同单词生成高相似度向量。

Normalizing Flow 是一种基于可逆变换的生成模型，其核心是将 BERT 输出的初始句子向量，转化为**平滑的各向同性高斯分布**。

> 高斯分布（正态分布）是一种连续概率分布，其概率密度函数呈钟形，由均值（μ）和方差（σ²）定义；在 LogGT 场景中，目标是让优化后的节点嵌入服从**标准各向同性高斯分布**（μ=0，σ²=1），其关键特性如下：
>
> 1. **平滑性**：分布的概率密度函数连续且可微，无突变或尖锐峰值。这意味着嵌入向量的数值变化平缓，不会因日志语法差异（如不同系统的字段顺序、表述风格）导致向量值剧烈波动，能有效过滤语法噪声带来的嵌入扰动🔶1-85。
> 2. **各向同性**：分布在向量空间的所有维度上具有相同的方差（即 “各维度特征独立且重要性一致”）。对于日志语义向量而言，这意味着向量的每个维度都均匀反映语义信息，不会因某一维度过度侧重语法格式（如特定系统的日志前缀）而掩盖核心语义，确保跨系统日志的语义特征在向量空间中 “公平表达”🔶1-85。

3.**时序感知图 Transformer（GTAT）**：

- 整合时间间隔特征：计算**日志事件的标准化时间间隔**，融入多头注意力机制，同时捕捉执行顺序与时间异常。
- 通过消息传递自动学习异构图高阶关系，输出图特征用于异常判断，AUC 值较传统序列模型提升超 2.3%。

4.**域适配迁移学习**：

- 设计语义加权方法：计算源 / 目标域日志语义与对方域平均语义的余弦相似度，生成注意力权重，突出高相似性语义。
- 结合最大均值差异（MMD）计算源 / 目标域图特征分布差异，以该差异为损失微调源域模型，实现无目标域标注数据的跨系统迁移。
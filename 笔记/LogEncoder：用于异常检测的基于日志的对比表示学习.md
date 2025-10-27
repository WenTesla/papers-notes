# LogEncoder：用于异常检测的基于日志的对比表示学习

https://www.doubao.com/thread/wf3877971ae69ca38



## 背景

**核心挑战**：

- **日志离散性**：日志是离散事件序列，难以捕捉事件间关联（不同于传感器等连续数据）；
- **类别不平衡**：正常日志占比极高（不平衡比 IR 常＞5），传统分类方法失效；<img src="https://cdn.xljsci.com/literature/176276523/page2/hizxye.png" alt="img" style="zoom:33%;" />
- **日志不稳定性**：开发者会修改源码新增 / 替换日志关键词（如 HDFS 日志中 “Interrupted” 改为 “terminating”）；
- **数据噪声**：日志采集 / 解析中易出现丢失、重复、错乱，影响检测性能。

## 现有方法的局限

- 监督方法需大量标注，成本高且受类别不平衡影响；
- 无监督 / 半监督方法无需标注，但因缺乏先验信息，检测精度低，对新日志事件鲁棒性差。



## 对比

![img](https://cdn.xljsci.com/literature/176276523/page2/esgyaj.png)



> 日志事件的核心价值不仅在于事件本身的标识，更在于事件描述中蕴含的语义关联（如 “Received block” 与 “Receiving block” 虽表述相近，但对应系统状态存在差异）。而事件嵌入技术仅将每个日志事件映射为一个固定向量，未深入解析事件文本中的关键词含义、语法结构及上下文逻辑，无法捕捉这种细粒度的语义信息。例如，对于 HDFS 日志中 “Interrupted” 改为 “terminating” 的场景（文档中提及的日志演化案例），事件嵌入会将二者视为完全不同的事件向量，无法识别其语义上均对应 “进程终止” 的核心含义，导致模型对相似事件的区分与关联能力不足
>
> word2vec 技术的应用逻辑是：先将日志事件拆分为单个单词，通过 word2vec 生成每个单词的向量，再采用加权平均（如结合 TF-IDF 权重）的方式合并为整个事件的向量。这种方式存在关键缺陷 —— 日志事件的语义是单词间**协同作用的整体**，而非单个单词的简单叠加。例如，日志事件 “Received block src: dest: of size” 中，“Received”“block”“size” 的语义关联（“接收指定大小的块数据”）无法通过加权平均保留，可能导致 “Received block” 与 “Sent block” 的向量差异被缩小（因核心单词 “block” 权重占比高），或 “of size” 等关键修饰信息的语义被稀释。
>
> 语义信息的丢失会使模型无法准确区分 “正常日志序列” 与 “异常日志序列” 的细微差异。例如，正常序列中 “Received block of size 100MB” 与异常序列中 “Received block of size 0MB”，若 word2vec 加权向量无法凸显 “100MB” 与 “0MB” 的语义差异，模型会将异常序列误判为正常；同时，这种语义模糊性还会导致模型学习到的 “正常模式” 边界模糊，进一步降低异常检测的精确率与召回率，与文档中提及的 “现有方法因语义丢失导致检测精度差” 的结论一致

## 贡献

我们提出了一个名为LogEncoder的日志异常检测框架，**其训练数据中仅包含正常日志**。因此，它不会受到有监督学习中类别不平衡问题的影响。

我们还使用预训练模型为离散的日志事件提取连续的语义向量，这克服了离散事件无法捕捉事件间相关性的挑战。

据我们所知，我们是首个提出在日志分析中采用单类分类并将对比学习作为模型目标的研究，这使得模型能够在学习可分离特征的同时保留序列的上下文信息。

我们进行了广泛的性能评估，以证明LogEncoder优于五种最先进的无监督和半监督学习方法。结果还表明，与有监督学习模型LogRobust相比，LogEncoder具有竞争力。





## 单类分类器

> 单类分类器是一类专门针对 “仅能获取大量正常数据，异常数据稀缺或难以标注” 场景的算法，其核心目标是：通过学习正常数据的分布模式，在 latent 空间中构建一个 “正常数据区域”，将异常数据与正常数据区分开 —— 即让所有正常数据映射到该区域内，异常数据因不符合正常模式而分布在区域外。
>
> 只能分类1和非1，非1具体是什么不知道

**传统核方法**：包括 One-Class Support Vector Machines（OCSVM）和 Support Vector Data Description（SVDD），其核心逻辑是通过 “核技巧” 将正常数据映射到高维空间，再构建一个包围正常数据的 “超球面” 或 “超平面” 作为正常区域边界

**深度学习扩展方法**：如 DeepSVDD 和 One-Class Neuron Network（OCNN），这类方法用神经网络替代传统核技巧，通过端到端训练学习正常数据的 latent 空间分布，并自动优化 “超球面” 的中心与半径



## 对比学习

对比学习能**显著提升下游任务性能**

对比学习的核心特征是**无需人工标注信息**，仅通过数据自身的内在特征（如相似性、差异性）完成训练，属于自监督学习范畴。

对比学习的核心逻辑是通过训练一个编码器f，让模型学会区分 “相似样本” 与 “不相似样本”，具体通过最大化特定目标实现：
$$
score\left(f(x), f\left(x^{+}\right)\right)>>score\left(f(x), f\left(x^{-}\right)\right)
$$
通过上述目标，对比学习强制编码器将相似样本的表征映射到 latent 空间的相近位置，将不相似样本的表征映射到远离位置，从而让模型学习到数据的 “判别性特征”。





## 框架

![img](https://cdn.xljsci.com/literature/176276523/page4/t2opjy.png)

### 日志嵌入

我们使用预训练模型为每个日志事件提取固定维度的语义向量。该模型将离散序列转换为连续序列，并减轻不稳定日志的影响。在本文中，我们选择Drain[20]方法将日志转换为日志事件和参数。接下来，我们使用预训练的BERT[30]提取每个日志事件的语义向量。

> 现有方法使用词嵌入技术（如Word2Vec[10]和FastText[31]）将日志事件中的关键词转换为向量，并通过日志事件中所有向量的加权和（如TF-IDF[32]）来表示日志事件。然而，使用关键词向量的加权和来表示日志事件可能会丢失有用的语义信息。

针对给定的正常日志事件序列\(S^{(i)}=(e_{1}^{(i)}, e_{2}^{(i)}, ..., e_{l}^{(i)}, ..., e_{L}^{i})\)（其中L为序列长度，\(e_{l}^{(i)}\)表示序列中第l个日志事件）。

#### 第一步：提取日志事件中的关键词（Tokenizer 处理）

首先通过**Tokenizer（分词器）** 对单个日志事件\(e_{l}^{(i)}\)进行处理，提取其中的关键词\(KW_{i}\)。这里的 Tokenizer 功能与 NLP 中的文本分词一致：将日志事件的完整文本（如 “*BLOCK NameSystem*addStoredBlock: blockMap updated: is added to*size*”）拆分为具有独立语义的基本单元（如 “BLOCK”“NameSystem”“addStoredBlock”“blockMap”“updated”“size” 等关键词）。

> 输入的是模板

#### 第二步：Transformer 编码器生成单事件语义向量

将第一步提取的关键词序列输入到**Transformer 编码器（文档中简称 TM）** 中，生成该日志事件\(e_{l}^{(i)}\)对应的固定维度语义向量\(z_{l}^{(i)}\)

#### 第三步：构建完整日志序列的语义向量序列

对日志事件序列\(S^{(i)}\)中的每个事件\(e_{1}^{(i)}, e_{2}^{(i)}, ..., e_{L}^{(i)}\)重复上述两步操作，分别生成对应的语义向量\(z_{1}^{(i)}, z_{2}^{(i)}, ..., z_{L}^{(i)}\)。这些向量按原日志事件的顺序组合，形成整个日志序列\(S^{(i)}\)在 “Log2Emb” 模块的输出 —— 语义向量序列\(Z^{(i)}=(z_{1}^{(i)}, z_{2}^{(i)}, ..., z_{l}^{(i)}, ..., z_{L}^{(i)})\)。该向量序列既保留了原日志序列的时序顺序，又通过连续语义向量解决了日志的离散性问题

个人疑惑1：

那为什么要模板解析，不直接输入每一条日志呢？

**原始日志包含大量冗余变量信息，会干扰语义提取与序列模式学习**

一、原始日志的 “变量 - 常量混合特性”：直接输入会导致语义混淆

原始日志由两部分构成：**常量部分（日志模板 / 日志事件，Log Event）** 与**变量部分（参数，Parameter）**。其中，常量部分是反映系统状态的核心语义载体（如 HDFS 日志中的 “Received block”“Verification succeeded”），而变量部分是动态变化的具体数值或标识（如块 ID “blk_160899987919862906”、IP 地址 “10.250.19.102”）。
若直接输入原始日志，模型会将变量部分的 “动态数值差异” 误判为 “语义差异”—— 例如，两条核心语义完全相同的日志 “Received block blk_A src: IP1” 与 “Received block blk_B src: IP2”，因块 ID 和 IP 不同，直接输入会被模型视为两个不同事件，导致语义表征分散，无法捕捉 “Received block” 这一统一系统行为模式。而日志解析（如使用 Drain 工具）可分离常量与变量，仅保留常量部分作为后续语义编码的输入，避免变量冗余信息干扰

二、适配 BERT 语义编码的 “输入一致性需求”：解析后才能实现标准化表征

LogEncoder 的 “Log2Emb” 模块依赖预训练 BERT 模型提取日志事件的语义向量，而 BERT 的有效语义学习需要**输入文本具备 “语义一致性”**—— 即相同系统行为对应的输入文本应统一，才能生成相似的语义向量。
原始日志中，同一系统行为的日志因变量不同会呈现多种文本形式（如 “Received block blk_X”“Received block blk_Y”），直接输入 BERT 会生成差异较大的语义向量，破坏 “相同行为语义相似” 的表征逻辑；而通过日志解析，可将这些原始日志统一映射到同一日志模板（如 “Received block <blk>”），消除变量干扰，使 BERT 能聚焦于常量部分的核心语义，生成具有一致性的 768 维向量，为后续序列模式学习奠定基础

三、保障后续 “序列模式学习” 的有效性：解析后才能构建有意义的日志序列

LogEncoder 的 “Emb2Rep” 模块需学习日志序列的正常模式（如 “接收块→验证→存储” 的事件顺序），而这一学习依赖 “以日志模板为基本单元的序列”。
若直接输入原始日志，序列中的每个元素都是 “包含变量的原始文本”，相同系统行为会被拆分为大量不同元素，导致序列模式碎片化（如 1000 条 “Received block + 不同块 ID” 的原始日志，会被视为 1000 种不同事件，无法形成 “Received block” 的高频正常模式）；而日志解析后，序列以 “日志模板” 为基本单元（如序列为 “Received block→Verification succeeded→addStoredBlock”），模型能清晰捕捉事件间的时序关联与正常模式，进而通过单类学习与对比学习区分正常 / 异常序列

### 嵌入到表示

<img src="https://cdn.xljsci.com/literature/176276523/page5/wegs3f.png" alt="img" style="zoom:67%;" />

1. **前置输入**：\(h_L\) 的输入是经 “Log2Emb” 模块生成的语义向量序列 \(Z^{(i)}=(z_1^{(i)}, z_2^{(i)}, ..., z_L^{(i)})\)（维度为 \((L, 768)\)，L 为日志序列长度，\(z_l^{(i)}\) 是单条日志模板的 BERT 语义向量）；
2. **中间处理**：\(Z^{(i)}\) 先通过 “多头注意力模块” 捕捉事件间的全局与局部关联（如 “Received block” 与 “Verification succeeded” 的因果关系），再输入到 LSTM 模块进行时序模式学习；
3. **\(h_L\) 的生成**：LSTM 会对序列进行 “逐事件” 的分步学习，每一步（对应第 l 个日志事件）都会输出一个隐藏状态 \(h_l\)，用于记录该步之前的序列上下文信息。其中，**最后一步隐藏状态 \(h_L\)** 是 LSTM 对整个序列（从第 1 个事件到第 L 个事件）所有上下文信息的融合结果，是对日志序列 “正常 / 异常模式” 的核心浓缩表征。

单分类
$$
{L}_{disc }=\min _{\Theta} \frac{1}{N} \sum_{i=1}^{N}\left\| x^{(i)}-c\right\| ^{2}+\lambda\| \Theta\| _{F}^{2}
$$

- N：训练集中正常日志序列的总数（如 HDFS 数据集中筛选的 6000 条正常日志）；
- \(x^{(i)}\)：第i条正常日志序列的最终表征向量（由 “Emb2Rep” 模块生成，维度如 128 维）；
- c：超球面的 “预定义球心”，文档中明确其初始化方式为 “所有正常序列表征向量的平均值”（即\(c=\frac{1}{N}\sum_{i=1}^N x^{(i)}\)），确保球心是正常序列特征的 “中心位置”；
- \(\left\| x^{(i)}-c\right\| ^{2}\)：第i条序列表征与球心的 “欧氏距离平方”，用于量化该序列与正常模式中心的偏离程度。

- \(\Theta\)：模型的所有可训练参数（如 “Emb2Rep” 模块中多头注意力的权重、LSTM 的门控参数、线性映射的w与b等）；
- \(\| \Theta\| _{F}^{2}\)：参数\(\Theta\)的 “Frobenius 范数平方”（即所有参数的平方和），用于衡量参数的整体规模；
- \(\lambda\)：超参数（如文档实验中可能取值 0.01、0.1 等），用于控制正则化项的强度 ——\(\lambda\)越大，对参数规模的约束越强。

对比学习

对比目标的核心思路是**自监督生成 “相似样本对” 与 “不相似样本对”**，通过优化目标让模型学会区分二者，进而捕捉序列的上下文信息与特征差异性。具体实现依赖 “日志序列掩码（Mask）” 技术

> 太难了，难以理解





![img](https://oss.xljsci.com//literature/176276523/page0/1758087152649.png)

通过小批量数据迭代训练，最小化 “单类损失（\(\mathcal{L}_{disc}\)）” 与 “对比损失（\(\mathcal{L}_{div}\)）” 的加权和（权重由超参数\(\alpha\)控制），最终让模型参数\(\Theta(f)\)收敛到最优状态 —— 即模型能将正常日志序列映射到 latent 空间的超球面内，同时保留序列的上下文信息，为后续异常检测提供可靠特征。

| 类型 |                           具体内容                           |                             作用                             |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 输入 | 1. 训练用正常日志事件序列集合\(\{S^{(i)}\}_{i=1}^{M}\)：\(S^{(i)}\)表示第i条正常日志序列，M为批量大小（batch size）2. 模型初始参数\(\Theta(f)\)：包括 Log2Emb 模块（BERT）、Emb2Rep 模块（注意力 + LSTM+MLP）的可训练参数3. 超参数\(\alpha\)：控制对比损失在总损失中的权重，平衡特征的判别性与多样性4. 迭代次数t：训练的总轮次，确保模型充分收敛5. 批量大小M：每轮训练用的样本数量，兼顾训练效率与参数更新稳定性 | 为训练提供数据、初始模型及关键控制参数，确保训练可启动且方向可控 |
| 输出 |              训练收敛后的模型参数\(\Theta(f)\)               |   用于后续日志序列的特征提取与异常检测（离线 / 在线检测）    |

AI写出的代码如下

```py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

# --------------------------
# 1. 模块定义：Log2Emb（日志到语义向量）
# --------------------------
class Log2Emb(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert.requires_grad_(True)  # 微调BERT
        
    def forward(self, log_events):
        """
        将日志事件序列转换为语义向量序列
        log_events: 单条日志事件序列，格式为列表，如 ["recv block", "send data", ...]
        return: 语义向量序列 (L, 768)，L为事件数量
        """
        semantic_vectors = []
        for event in log_events:
            # BERT编码单个日志事件
            inputs = self.tokenizer(event, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():  # 若固定BERT权重，可加此句
                outputs = self.bert(**inputs)
            # 取[CLS] token的输出作为事件语义向量
            event_vec = outputs.last_hidden_state[:, 0, :].squeeze(0)  # (768,)
            semantic_vectors.append(event_vec)
        return torch.stack(semantic_vectors, dim=0)  # (L, 768)


# --------------------------
# 2. 模块定义：Emb2Rep（嵌入到表示）
# --------------------------
class Emb2Rep(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=4, batch_first=True
        )
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, 
            num_layers=2, batch_first=True, bidirectional=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 双向LSTM输出需×2
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z_seq):
        """
        将语义向量序列转换为latent表示
        z_seq: 批量语义向量序列 (batch_size, L, 768)
        return: latent表示 (batch_size, output_dim)
        """
        # 多头注意力捕捉全局依赖
        attn_output, _ = self.multihead_attn(z_seq, z_seq, z_seq)  # (batch_size, L, 768)
        
        # LSTM捕捉时序特征
        lstm_output, _ = self.lstm(attn_output)  # (batch_size, L, 2*hidden_dim)
        
        # 取最后一个时间步输出，经MLP映射到latent空间
        last_step_output = lstm_output[:, -1, :]  # (batch_size, 2*hidden_dim)
        x = self.mlp(last_step_output)  # (batch_size, output_dim)
        return x


# --------------------------
# 3. 损失函数定义
# --------------------------
class LogEncoderLoss(nn.Module):
    def __init__(self, center, lambda_reg=1e-5):
        super().__init__()
        self.center = center  # 超球面中心 (output_dim,)
        self.lambda_reg = lambda_reg  # 正则化系数
        
    def forward(self, x, x_mask, alpha=0.5):
        """
        联合损失：单类损失 + alpha×对比损失
        x: 原始序列latent表示 (batch_size, output_dim)
        x_mask: 掩码序列latent表示 (batch_size, output_dim)
        alpha: 对比损失权重
        """
        # 1. 单类损失 L_disc：最小化正常样本到超球面中心的距离
        dist_to_center = torch.norm(x - self.center, dim=1)  # (batch_size,)
        l_disc = torch.mean(dist_to_center** 2)  # 平均距离平方
        
        # 2. 对比损失 L_div：最大化原始与掩码序列的相似度
        cos_sim = nn.functional.cosine_similarity(x, x_mask, dim=1)  # (batch_size,)
        l_div = -torch.mean(cos_sim)  # 负余弦相似度（需最小化）
        
        # 3. 总损失（含L2正则化）
        total_loss = l_disc + alpha * l_div
        # 添加模型参数正则化（防止过拟合）
        reg_loss = sum(p.norm()**2 for p in self.parameters()) * self.lambda_reg
        return total_loss + reg_loss


# --------------------------
# 4. 训练流程（Algorithm 1实现）
# --------------------------
def train_logencoder(normal_sequences,  # 正常日志序列集合，格式：list[list[str]]
                     t=100,            # 迭代次数
                     batch_size=32,    # 批量大小
                     alpha=0.5,        # 对比损失权重
                     output_dim=128):  # latent表示维度
    
    # 初始化模块
    log2emb = Log2Emb()
    emb2rep = Emb2Rep(output_dim=output_dim)
    # 初始化超球面中心（用首批数据的平均表示）
    init_batch = normal_sequences[:batch_size]
    init_z = [log2emb(seq).unsqueeze(0) for seq in init_batch]  # 每个样本：(1, L, 768)
    init_z_batch = torch.cat(init_z, dim=0)  # (batch_size, L, 768)
    init_x = emb2rep(init_z_batch)  # (batch_size, output_dim)
    center = torch.mean(init_x, dim=0).detach()  # 固定初始中心
    
    # 损失函数与优化器
    criterion = LogEncoderLoss(center)
    optimizer = optim.Adam(
        list(log2emb.parameters()) + list(emb2rep.parameters()),
        lr=1e-4
    )
    
    # Algorithm 1: 小批量SGD训练
    for epoch in range(t):  # 步骤1：总迭代次数t
        # 随机打乱数据并分批次
        batch_indices = torch.randperm(len(normal_sequences)).split(batch_size)
        
        for batch_idx in batch_indices:  # 遍历每个批次
            batch_seqs = [normal_sequences[i] for i in batch_idx]  # 步骤2：取批量数据
            batch_z = []
            batch_x = []
            batch_x_mask = []
            
            for seq in batch_seqs:  # 步骤2：处理单条序列
                # 步骤3：Log2Emb生成语义向量序列Z^(i)
                z = log2emb(seq)  # (L, 768)
                batch_z.append(z.unsqueeze(0))  # 加batch维度
            
            # 批量处理：原始序列特征x^(i)
            z_batch = torch.cat(batch_z, dim=0)  # (batch_size, L, 768)
            x = emb2rep(z_batch)  # 步骤4：生成x^(i)
            
            # 步骤5：生成掩码序列Z_mask^(i)
            mask_prob = 0.2  # 随机掩码20%的事件向量
            z_mask_batch = z_batch * (torch.rand(z_batch.shape) > mask_prob).float()  # 掩码操作
            
            # 步骤7：生成掩码序列特征x_mask^(i)
            x_mask = emb2rep(z_mask_batch)
            
            # 步骤8：计算损失并更新参数
            loss = criterion(x, x_mask, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{t}], Loss: {loss.item():.4f}")
    
    # 步骤10：返回训练好的模型参数
    return {
        'log2emb': log2emb.state_dict(),
        'emb2rep': emb2rep.state_dict(),
        'center': center
    }


# --------------------------
# 示例：模拟数据训练
# --------------------------
if __name__ == "__main__":
    # 模拟正常日志序列（每条序列含3-5个事件）
    normal_sequences = [
        ["recv block from src", "write block to disk", "close connection"],
        ["send heartbeat signal", "recv ack", "update status"],
        # ... 更多正常序列 ...
    ] * 100  # 扩充数据量
    
    # 训练模型
    trained_params = train_logencoder(
        normal_sequences,
        t=50,
        batch_size=16,
        alpha=0.3
    )
    print("训练完成，参数已保存")
```

- `Log2Emb`类实现步骤 3 的语义向量生成（基于 BERT）；
- `Emb2Rep`类实现步骤 4 和 7 的 latent 表示学习（含多头注意力 + LSTM+MLP）；
- `LogEncoderLoss`类实现步骤 8 的联合损失计算（\(\mathcal{L}_{disc}+\alpha\mathcal{L}_{div}\)）。

步骤 5 通过随机将 20% 的事件向量置零实现，模拟日志噪声或信息缺失。

**超球面中心**：初始化为首批数据的平均 latent 表示，训练中固定（仅优化模型参数使正常样本向中心聚集）。

**训练逻辑**：严格遵循 Algorithm 1 的迭代流程，外层控制总轮次、内层处理批量数据，通过梯度下降更新参数。





### 异常检测

前一阶段学习到的特征保留了上下文信息，正常序列分布在潜在空间的超球面中，这与异常检测阶段是解耦的，离线检测是对已生成的日志事件序列进行异常检测，将日志事件序列的表示与c之间的距离作为异常分数，以此判断该序列是否为异常序列。在线检测则基于历史日志事件序列预测即将发生的事件，并将其与实际发生的事件进行比较。 

#### 离线检测

$$
A\left(x^{(i)}\right)=\left\| x^{(i)}-c\right\| ^{2}
$$

$$
label =\left\{\begin{array}{l} Normal if A\left(x^{(i)}\right)<\varepsilon \\ Abnormal else \end{array}\right.
$$

#### 在线检测

**历史序列滑动更新**：以 “滑动窗口” 或 “会话窗口”（如 HDFS 中的 “blockID” 关联的序列）为单位，实时维护最新的历史日志序列\(S_{real}\)（如最近 20 条日志组成的序列

**生成实时表征\(x_{real}\)**：将\(S_{real}\)输入 Log2Emb（生成语义向量序列）与 Emb2Rep（生成序列表征），得到实时上下文特征\(x_{real}\)；

**输出下一个事件概率\(p_{real}\)**：将\(x_{real}\)输入预测模型\(P(x)\)，得到K维概率向量\(p_{real}\)，其中概率最高的前k个事件（如 top@1、top@10）被视为 “最可能发生的下一个事件候选集”。

## 结果

![img](https://oss.xljsci.com//literature/176276523/page0/1758089006268.png)






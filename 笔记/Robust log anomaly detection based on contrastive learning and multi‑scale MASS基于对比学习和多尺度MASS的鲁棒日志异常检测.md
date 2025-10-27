# Robust log anomaly detection based on contrastive learning and multi‑scale MASS基于对比学习和多尺度MASS的鲁棒日志异常检测

## 前提

日志易因操作人员操作随意或模板更新产生噪声（如同义词替换、增删词），多数方法将此类正常噪声误判为异常，鲁棒性差。

多数机器学习 / 深度学习方法（如 IM、LR、DeepLog）未解决日志噪声问题，DeepLog 虽考虑噪声但需用户反馈增量更新，LogRobust 未充分学习多尺度上下文导致检测精度有限；



## 设计

![img](https://cdn.xljsci.com/literature/183079254/page5/t1b0a7.png)

1. **日志解析**：采用 Drain 算法，将非结构化日志消息转换为结构化日志模板（分离常量字符串与变量值），为后续特征提取奠定基础。
2. **鲁棒语义提取（对比学习 + 改进 BERT）**：
   - 将对比学习融入 BERT 模型，通过对比损失函数缩小正常日志模板（含正常噪声模板）间语义距离、扩大正常与异常模板距离，确保正常模板语义相似度更高；
   - 设定余弦相似度阈值 S，对新日志模板分类：若与现有正常模板相似度均超 S 则归为最相似类别，否则判定为异常，避免将正常噪声误判。
3. **多尺度 MASS（MSMASS）异常检测**：
   - 改进 MASS 模型，将多头自注意力机制替换为多尺度多头自注意力（MSAttention），可同时捕捉日志序列的局部与全局上下文信息；
   - 训练时对正常日志序列进行连续掩码，通过 Transformer 的 Encoder-Decoder 结构预测掩码模板；检测时若真实模板不在预测结果 Top-K 内，则判定序列异常。

### 语义提取

![img](https://cdn.xljsci.com/literature/183079254/page8/dx0f0l.png)

#### 1. 输入部分：三类日志序列片段

- **seg₁**：由**原始正常日志模板**组成的序列片段（如\(k_1, k_2, k_3, k_4\)），代表无噪声的正常日志模式。
- **seg₂**：由**噪声注入后的正常日志模板**组成的序列片段（如\(k_1, k_1^*, k_3, k_4\)，其中\(k_1^*\)是\(k_1\)的同义词替换或词语增删版本），模拟真实场景中正常日志的 “噪声干扰”。
- **seg₃**：另一类**噪声注入后的正常日志模板**组成的序列片段（如\(k_1^*, k_1, k_3, k_4\)），进一步丰富正常噪声的多样性。

loss如下
$$
l_{ni}=-(log(\frac{e^{sim(h_{ni},h^+_{ni})/\tau}}{e^{sim(h_{ni,h^+_{ni}})}+\sum_{j=1}^le^{sim(h_{ni},h_{aj})/\tau}})+log(\frac{e^{sim(h_{ni},h_{ni}^*)/\tau}}{e^{sim(h_{ni},h_{ni}^*)/\tau}+\sum_{j=1}^le^{sim(h_{ni},h_{aj})/\tau}})) \\
l_{ni}^*=-log(\frac{e^{sim(h_{ni}^*,h_{ni})/\tau}}{e^{sim(h_{ni}^*,h_{ni})/\tau}+\sum_{j=1}^le^{sim(h_{ni^*,h_{aj}})/\tau)}}) \\
$$
正样本对是 “单个原始正常模板的\(h_{ni}\)” 与 “单个正常噪声模板的\(h_{ni}^*\)”“单个关联正常模板的\(h_{ni}^+\)”；

负样本是 “单个正常模板的\(h_{ni}\)” 与 “所有单个异常模板的\(h_{aj}\)”

模型通过优化这些**单个向量间的相似度**（而非序列整体向量的相似度），实现 “正常模板聚类、异常模板分离” 的目标。

1. 以序列为输入（为单个模板提供上下文），但 BERT 最终编码**单个日志模板k**，输出其语义向量h；
2. 对比学习通过**单个h的组合**（正样本对、负样本对）计算损失，优化语义空间中向量的分布

> \(\tau\)（温度系数）：可调节超参数，用于控制模型对语义相似度差异的 “敏感度”——\(\tau\)越小，模型对相似模板与不相似模板的区分度越高；\(\tau\)过大则可能导致模型无法有效区分负样本（异常模板）间的差异，文档中最终将其设为 1.0 以避免对分类产生额外干扰。
>
> sim函数是余弦相似度

### **多尺度 MASS（MSMASS）异常检测**

改进 MASS 模型，将多头自注意力机制替换为多尺度多头自注意力（MSAttention），可同时捕捉日志序列的局部与全局上下文信息；

训练时对正常日志序列进行连续掩码，通过 Transformer 的 Encoder-Decoder 结构预测掩码模板；检测时若真实模板不在预测结果 Top-K 内，则判定序列异常。

**MSMASS（Multi-scale Masked Sequence Sequence to Sequence）** 是基于改进的掩码序列到序列模型，核心是通过多尺度注意力机制捕捉日志序列的局部与全局上下文信息，实现对正常日志序列模式的精准学习，为异常检测提供关键支撑。

1. **多尺度划分**：为每个注意力头分配特定的 “尺度”（即关注的日志序列片段长度），例如在 HDFS 数据集上设置尺度为 1、3、5、16（分别对应单个模板、连续 3 个模板、连续 5 个模板、整个序列）。
2. **差异化关注**：
   - 小尺度注意力头（如尺度 1、3）聚焦于局部上下文，捕捉相邻日志模板的直接关联（如 “接收数据块” 后紧跟 “写入磁盘” 的逻辑）；
   - 大尺度注意力头（如尺度 5、16）关注全局上下文，学习整个序列的整体模式（如系统启动流程的完整日志序列规律）。



#### 1. 训练阶段（基于正常日志序列）

遵循 “掩码 - 预测” 的自监督学习模式（对应 Algorithm 1）：

- **输入**：正常日志序列集合X（由日志模板组成的有序序列）。
- **掩码操作**：对每条序列x，选取长度为M（序列长度的 50%）的连续片段进行掩码（替换为 [MASK]），得到掩码序列\(x^{\backslash u:v}\)（Encoder 输入）和反向掩码序列\(x''\)（Decoder 输入）。
- **编码与解码**：
  - Encoder 对\(x^{\backslash u:v}\)编码，输出未掩码部分的上下文特征；
  - Decoder 结合 Encoder 输出和\(x''\)，预测被掩码的片段（目标是还原真实日志模板）。
- **损失优化**：通过对数似然损失函数，最小化预测模板与真实模板的差异，使模型学习正常序列的多尺度上下文模式。

#### 2. 检测阶段（判断新日志序列是否异常）

- 对输入的新日志序列，随机选取连续片段进行掩码，输入训练好的 MSMASS 模型；
- 模型预测被掩码的模板，若真实模板不在预测结果的 Top-K 列表中（K 为预设阈值，如 HDFS 设为 5），则判定该序列为异常（不符合正常模式）。



豆包写的代码如下

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --------------------------
# 1. 多尺度注意力层（MSAttention）
# --------------------------
class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, num_heads, scales=[1,3,5,16]):
        """
        多尺度多头自注意力层
        :param d_model: 模型维度（如512）
        :param num_heads: 注意力头总数
        :param scales: 关注的序列尺度（片段长度）列表
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.scales = scales
        self.head_dim = d_model // num_heads
        
        # 按尺度分配注意力头（文档中按"每层各尺度均等"原则）
        self.heads_per_scale = self._allocate_heads()
        
        # 线性层：用于计算Q、K、V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # 输出投影层
        self.out_proj = nn.Linear(d_model, d_model)

    def _allocate_heads(self):
        """按尺度分配注意力头数量（示例：均等分配）"""
        base = self.num_heads // len(self.scales)
        remainder = self.num_heads % len(self.scales)
        heads = [base + 1 if i < remainder else base for i in range(len(self.scales))]
        return dict(zip(self.scales, heads))

    def forward(self, x, mask=None):
        """
        :param x: 输入序列，shape=(batch_size, seq_len, d_model)
        :param mask: 掩码矩阵（可选），shape=(batch_size, seq_len, seq_len)
        :return: 输出序列，shape=(batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算Q、K、V：(batch_size, seq_len, 3*d_model) → 拆分为3个(d_model)
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  # 每个元素shape=(batch_size, seq_len, d_model)
        q, k, v = [t.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
                   for t in qkv]  # 每个shape=(batch_size, num_heads, seq_len, head_dim)
        
        # 按尺度处理不同注意力头
        head_outputs = []
        current_head = 0
        for scale in self.scales:
            num_heads = self.heads_per_scale[scale]
            if num_heads == 0:
                continue
            
            # 截取当前尺度对应的注意力头
            q_scale = q[:, current_head:current_head+num_heads]
            k_scale = k[:, current_head:current_head+num_heads]
            v_scale = v[:, current_head:current_head+num_heads]
            
            # 尺度掩码：仅关注当前尺度内的上下文（局部/全局）
            if scale < seq_len:
                # 局部尺度：仅允许关注当前位置±scale//2范围内的元素
                scale_mask = torch.ones_like(mask[:, :seq_len, :seq_len]) if mask is not None else None
                for i in range(seq_len):
                    start = max(0, i - scale//2)
                    end = min(seq_len, i + scale//2 + 1)
                    if scale_mask is not None:
                        scale_mask[:, :, i, :start] = 0
                        scale_mask[:, :, i, end:] = 0
            else:
                # 全局尺度：关注整个序列
                scale_mask = mask
            
            # 计算注意力分数
            attn_scores = (q_scale @ k_scale.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, seq_len, seq_len)
            if scale_mask is not None:
                attn_scores = attn_scores.masked_fill(scale_mask == 0, -1e9)
            attn_probs = F.softmax(attn_scores, dim=-1)
            
            # 注意力加权求和
            output = attn_probs @ v_scale  # (batch, heads, seq_len, head_dim)
            head_outputs.append(output)
            
            current_head += num_heads
        
        # 拼接所有头的输出并投影
        all_heads = torch.cat(head_outputs, dim=1)  # (batch, num_heads, seq_len, head_dim)
        all_heads = all_heads.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        return self.out_proj(all_heads)


# --------------------------
# 2. MSMASS模型（Encoder-Decoder）
# --------------------------
class MSMASS(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=3, scales=[1,3,5,16]):
        """
        多尺度掩码序列到序列模型
        :param vocab_size: 日志模板词汇表大小
        :param d_model: 模型维度
        :param num_heads: 注意力头总数
        :param num_layers: Encoder/Decoder层数
        :param scales: 多尺度列表
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 嵌入层（日志模板→向量）
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码（加入时序信息）
        self.pos_encoding = self._generate_pos_encoding(max_len=100, d_model=d_model)
        
        # Encoder层（多尺度注意力+前馈网络）
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=2048,
                custom_attention=MultiScaleAttention(d_model, num_heads, scales)  # 替换为多尺度注意力
            ) for _ in range(num_layers)
        ])
        
        # Decoder层（同上）
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=2048,
                custom_attention=MultiScaleAttention(d_model, num_heads, scales)
            ) for _ in range(num_layers)
        ])
        
        # 输出层（预测日志模板）
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _generate_pos_encoding(self, max_len, d_model):
        """生成位置编码（正弦余弦函数）"""
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)
        return pe  # (max_len, 1, d_model)

    def encode(self, src):
        """Encoder：输入掩码序列，输出上下文特征"""
        batch_size, seq_len = src.shape
        # 嵌入+位置编码
        x = self.embedding(src) + self.pos_encoding[:seq_len].permute(1, 0, 2)  # (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # Transformer要求输入为(seq_len, batch, d_model)
        
        # 多层Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        return x  # (seq_len, batch, d_model)

    def decode(self, tgt, memory):
        """Decoder：输入反向掩码序列+Encoder输出，预测掩码模板"""
        batch_size, seq_len = tgt.shape
        x = self.embedding(tgt) + self.pos_encoding[:seq_len].permute(1, 0, 2)
        x = x.transpose(0, 1)
        
        # 多层Decoder
        for layer in self.decoder_layers:
            x = layer(x, memory)
        return x.transpose(0, 1)  # (batch, seq_len, d_model)

    def forward(self, src, tgt):
        """
        :param src: Encoder输入（掩码序列），shape=(batch, seq_len)
        :param tgt: Decoder输入（反向掩码序列），shape=(batch, seq_len)
        :return: 预测的模板分布，shape=(batch, seq_len, vocab_size)
        """
        memory = self.encode(src)  # (seq_len, batch, d_model)
        dec_output = self.decode(tgt, memory)  # (batch, seq_len, d_model)
        return self.output_proj(dec_output)  # (batch, seq_len, vocab_size)


# --------------------------
# 3. 训练流程（基于Algorithm 1）
# --------------------------
class LogDataset(Dataset):
    """日志序列数据集（示例）"""
    def __init__(self, normal_sequences, seq_len=16, mask_ratio=0.5):
        """
        :param normal_sequences: 正常日志序列列表（每个元素是模板ID序列）
        :param seq_len: 序列长度
        :param mask_ratio: 掩码比例（文档中为50%）
        """
        self.sequences = [seq[:seq_len] for seq in normal_sequences if len(seq)>=seq_len]
        self.seq_len = seq_len
        self.mask_len = int(seq_len * mask_ratio)
        self.mask_id = max([max(seq) for seq in self.sequences]) + 1  # 掩码符号ID

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 随机选择掩码起始位置u（文档中步骤1-2）
        u = np.random.randint(0, self.seq_len - self.mask_len + 1)
        v = u + self.mask_len  # 掩码结束位置
        
        # 构造Encoder输入：掩码u~v区域（步骤6）
        src = seq.copy()
        src[u:v] = [self.mask_id] * self.mask_len
        
        # 构造Decoder输入：掩码非u~v区域（步骤8）
        tgt = seq.copy()
        tgt[:u] = [self.mask_id] * u
        tgt[v:] = [self.mask_id] * (self.seq_len - v)
        
        # 目标序列：u~v区域的真实模板（用于计算损失）
        target = [0] * self.seq_len
        target[u:v] = seq[u:v]  # 仅对掩码区域计算损失
        return torch.tensor(src), torch.tensor(tgt), torch.tensor(target)


def train_msmass():
    # 超参数
    vocab_size = 100  # 日志模板总数（示例）
    seq_len = 16      # 序列长度（如HDFS数据集）
    epochs = 10
    batch_size = 32
    lr = 1e-4
    
    # 生成模拟数据（正常日志序列）
    normal_sequences = [np.random.randint(0, vocab_size, size=seq_len) for _ in range(1000)]
    dataset = LogDataset(normal_sequences, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型、损失函数、优化器
    model = MSMASS(vocab_size=vocab_size, scales=[1,3,5,16])
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略非掩码区域的损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环（文档中步骤3-12）
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for src, tgt, target in dataloader:
            # 前向传播
            pred = model(src, tgt)  # (batch, seq_len, vocab_size)
            loss = criterion(pred.reshape(-1, vocab_size), target.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    return model


# 运行训练
if __name__ == "__main__":
    trained_model = train_msmass()
```

算法如下

![img](https://oss.xljsci.com//literature/183079254/page0/1760944000841.png)



通用的MASS代码如下

```python
import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(MSAA, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        # # x2 是从低到高，x4是从高到低的设计，x2传递语义信息，x4传递边缘问题特征补充
        # x_1_2_fusion = self.fusion_1x2(x1, x2)
        # x_1_4_fusion = self.fusion_1x4(x1, x4)
        # x_fused = x_1_2_fusion + x_1_4_fusion
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out


if __name__ == '__main__':
    x = torch.randn(4, 64, 128, 128).cuda()
    y = torch.randn(4, 64, 128, 128).cuda()
    z = torch.randn(4, 64, 128, 128).cuda()
    model = MSAA(192, 64).cuda()
    out = model(x, y, z)
    print(out.shape)
```

[(99+ 封私信 / 52 条消息) (即插即用模块-特征处理部分) 四十一、(2024) MSAA 多尺度注意力聚合模块 - 知乎](https://zhuanlan.zhihu.com/p/1898488179352371421)

## 结果

按时间顺序8：2划分



![img](https://cdn.xljsci.com/literature/183079254/page18/3vrlct.png)

![img](https://cdn.xljsci.com/literature/183079254/page18/g9ayuh.png)

![img](https://cdn.xljsci.com/literature/183079254/page19/ug5o8x.png)

![img](https://cdn.xljsci.com/literature/183079254/page15/gqezas.png)

- **坐标轴**：横轴和纵轴均标注为 “a1 - a24”，代表日志模板的类别（如 HDFS 数据集有 30 类模板，此处选取前 24 类进行可视化；BGL 数据集有 377 类模板，同样选取前 24 类）。
- **颜色刻度**：右侧颜色条从浅黄色（相似度≈0）到深蓝色（相似度≈1），颜色越深表示两个模板的语义相似度越高。

a是abnormal异常的，可以看出正常之间相似度高，正常与异常的相似度低。


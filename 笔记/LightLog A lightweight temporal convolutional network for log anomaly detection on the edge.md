# LightLog: A lightweight temporal convolutional network for log anomaly detection on the edge



开源地址：[Aquariuaa/LightLog: An deep learning based lightweight TCN for log anomaly detection.](https://github.com/Aquariuaa/LightLog)

## 贡献

• 我们提出了一种适用于**边缘设备实时处理**的日志异常检测算法。该方法避免将大规模日志文件上传至网络服务器进行分析，从而实现**本地实时检测处理**，提高了边缘设备的可靠性和安全性。（==轻量化==）

• 我们提出了一种新的映射方法，可显著降低日志模板转换后的语义向量维度。该方法解决了现有方法中使用独热编码时忽略模板语义相似性的问题。此外，它还减轻了训练和检测过程中下游处理管道的巨大计算负担。(==好像有人已经解决了这个问题==)

• 我们设计了一种轻量级时间卷积网络（TCN），用于时间日志数据的有监督分类。该模型采用多核逐点卷积来改进TCN中的残差块，并通过全局平均池化减少模型的参数数量。这使得算法的浮点运算次数（FLOPs）、参数数量、检测模型大小和检测时间都显著减少，同时提高了检测精度。





## 设计

![img](https://cdn.xljsci.com/literature/174597327/page3/a56ux7.png)

### 训练

#### 1. 第一步：日志数据的序列化与传输（从边缘到云端）

训练的初始环节是获取边缘设备的日志数据并规范传输格式：

- **数据来源**：收集边缘设备（如物联网终端、边缘计算节点）生成的原始日志；
- **序列化处理**：按 “滑动窗口” 或 “会话窗口” 规则对日志进行序列化 —— 滑动窗口以固定日志条数为单位（如窗口大小 300、步长 100，适配无固定标识的日志，如 BGL 数据集），会话窗口则基于唯一标识（如日志的块 ID、线程号、任务号，适配有明确关联的日志，如 HDFS 数据集的块 ID 关联日志）；
- **传输目标**：将序列化后的日志数据传输至云端计算平台，利用云端高性能算力完成后续训练（边缘设备算力有限，仅负责部署检测，不承担训练任务）。

#### 2. 第二步：非结构化日志的特征转化（Mapping 组件核心作用）

此步骤将杂乱的原始日志转化为可用于模型训练的 “低维语义向量”，解决传统日志特征维度高、语义丢失的问题：

- **第一步：提取日志模板**：先对非结构化原始日志进行解析，分离 “日志模板”（固定文本结构，如 “Receiving block <*> src: /<*> dest: /<*>”）和 “事件参数”（可变内容，如具体块 ID、IP 地址），聚焦模板分析（参数易变，对异常检测无核心意义）；
- **第二步：生成高维语义向量**：用 Word2Vec 算法将日志模板转化为 300 维语义向量 —— 相比 One-Hot 编码（无语义信息、向量稀疏），Word2Vec 能保留模板间的语义相似性（如 “接收块” 与 “传输块” 的模板向量更接近），提升特征表达能力；
- **第三步：PCA-PPA 降维**：通过 Mapping 组件的 PCA-PPA 算法对 300 维高维向量降维：先经 PCA（主成分分析）提取核心特征，再用 PPA（后处理算法）去除 PCA 空间中的均值向量和主导方向（超参数 d 默认设为 7，控制去除的特征方向数量），最终得到低维语义向量；
- **关键价值**：既解决了高维向量导致的下游计算负担问题，又保留了模板的核心语义信息，同时建立 “日志模板 - 低维语义向量” 的可维护映射关系 —— 当出现新日志模板时，可在云端重新构建映射，确保模型适应性。

#### 3. 第三步：轻量级 TCN 模型的训练（原始 TCN 的优化与训练）

基于低维语义向量，对原始 TCN 进行改进并训练，确保模型在高精度基础上实现轻量化：

- **TCN 的核心改进**：针对边缘设备算力限制，对原始 TCN 做两项关键优化：
  - 引入 “多核点卷积”：在 TCN 的残差块中加入多组卷积核（如 1×3、1×5），通过点卷积（1×k 大小的卷积核）对多通道特征图加权融合，既丰富日志序列的特征提取维度（捕捉不同类型的异常模式），又减少卷积参数数量；
  - 用 “全局平均池化（GAP）替代全连接层”：对 TCN 最后一层卷积输出的特征图做全局平均计算，直接生成紧凑特征向量用于分类，彻底消除全连接层的冗余参数，同时缓解过拟合；
- **模型训练目标**：以低维语义向量为输入，训练 TCN 完成 “日志序列正常 / 异常” 的二分类任务，优化目标为降低二进制交叉熵损失（适配二分类场景），采用 Adam 优化器（初始学习率 0.001，每 20 轮按 0.98 衰减），训练 100 轮，最终得到轻量级 TCN 检测模型；
- **训练的核心意义**：借助云端算力完成模型训练，确保模型在 HDFS、BGL 等数据集上达到高检测精度（后续实验验证 F1 分数分别达 97.0%、97.2%），同时通过优化将模型参数压缩至 544 个、大小仅 139KB，为边缘设备部署奠定基础。

### 部署

- **映射函数**：即云端训练阶段建立的 “日志模板 - 低维语义向量” 映射关系（经 Word2Vec+PCA-PPA 降维得到）。当边缘设备生成新日志时，该函数能快速将日志解析后的模板，转化为模型可识别的低维语义向量，避免本地进行高耗算的特征降维操作；
- **增强型 TCN 模型**：即云端训练好的轻量化时序卷积网络（经多核点卷积、全局平均池化优化），模型大小仅 139KB、参数仅 544 个，完全适配边缘设备有限的内存与算力（如 Nvidia Jetson TX2 这类边缘硬件）。

### 日志处理

<img src="https://cdn.xljsci.com/literature/174597327/page4/v6etgs.png" alt="img" style="zoom: 67%;" />

降维（向量维度太高）

```sh
# 函数功能：将日志模板的高维语义向量（Word2Vec生成）降维为低维语义向量
# 输入：
#   high_dim_vecs: 高维语义向量集合（类型：列表/数组，元素为单条日志模板的300维向量，如[vec1, vec2, ..., vecN]）
#   d: 超参数（整数，控制需去除的主导特征方向数量，文档默认d=7）
# 输出：
#   low_dim_vecs: 低维语义向量集合（类型：列表/数组，元素为降维后的向量）
# 依赖：需提前实现PCA降维函数（输入向量集合，输出降维后的向量集合）

Function generate_low_dim_semantic_vec(high_dim_vecs, d):
    # 步骤1：对高维语义向量执行第一次PCA降维，初步去除低方差噪声
    pca1_vecs = PCA(high_dim_vecs)
    
    # 步骤2：中心化处理——消除全局数值偏移，使向量均值为0
    # 计算pca1_vecs中所有向量的“均值向量”（每个维度取所有样本的平均值）
    avg_vec = calculate_average_vector(pca1_vecs)
    # 每个向量减去均值向量
    centered_vecs = [vec - avg_vec for vec in pca1_vecs]
    
    # 步骤3：对中心化后的向量执行第二次PCA降维，强化核心语义特征
    pca2_vecs = PCA(centered_vecs)
    
    # 步骤4：遍历每个样本向量，通过PPA算法去除主导特征冗余
    low_dim_vecs = []
    for v in pca2_vecs:  # v：单条日志模板经两次PCA后的向量
        # 步骤5：计算v在“前d个主导特征方向”上的总投影，并从v中减去
        total_projection = 0  # 初始化“总投影向量”
        # 取pca2_vecs中的前d个向量（即方差最大的d个主导特征方向）
        for n in 1 to d:
            # 取第n个主导特征方向的基向量（pca2_vecs中的第n个向量）
            base_vec = pca2_vecs[n-1]  # 若索引从0开始，需调整为n-1
            # 计算v与base_vec的内积（投影系数，反映v对该特征方向的依赖程度）
            dot_product = calculate_dot_product(v, base_vec)
            # 计算v在该基向量上的投影向量（投影系数 × 基向量）
            projection = dot_product * base_vec
            # 累加投影向量，得到总投影
            total_projection = total_projection + projection
        # 从原向量v中减去总投影，得到低维向量
        low_dim_vec = v - total_projection
        # 将结果加入低维向量集合
        low_dim_vecs.append(low_dim_vec)
    
    # 步骤6：返回最终的低维语义向量集合
    return low_dim_vecs


# 辅助函数1：计算向量集合的均值向量
Function calculate_average_vector(vec_collection):
    # 获取向量维度（如第一次PCA后向量的维度k）
    dim = length(vec_collection[0])
    # 初始化均值向量（各维度初始值为0）
    avg_vec = [0 for _ in 1 to dim]
    # 遍历每个向量，累加各维度数值
    for vec in vec_collection:
        for i in 1 to dim:
            avg_vec[i-1] = avg_vec[i-1] + vec[i-1]  # 索引从0开始时调整
    # 各维度数值除以向量总数，得到均值
    num_vecs = length(vec_collection)
    avg_vec = [val / num_vecs for val in avg_vec]
    return avg_vec


# 辅助函数2：计算两个向量的内积（假设向量维度相同）
Function calculate_dot_product(vec_a, vec_b):
    dot_sum = 0
    dim = length(vec_a)
    for i in 1 to dim:
        dot_sum = dot_sum + (vec_a[i-1] * vec_b[i-1])  # 索引从0开始时调整
    return dot_sum
```

> **PPA（后处理算法，Post Processing Algorithm）** 是与 PCA 协同工作的关键技术，核心作用是在 PCA 初步降维的基础上进一步优化特征表达 —— 既保留关键语义信息，又通过去除冗余特征实现更紧凑的向量表示，为后续轻量级 TCN 模型适配边缘设备奠定基础

在 LightLog 中，PPA 的处理对象是 “Word2Vec 生成的 300 维日志模板语义向量经 PCA 降维后的中间向量”，最终输出更紧凑的低维语义向量，直接作为 TCN 模型的输入。

## 实验

### 数据集

![img](https://cdn.xljsci.com/literature/174597327/page5/vca3v1.png)

> 训练数据这少？

BGL滑动窗口

HDFS会话块

==如果序列中有任何错误的日志，则将日志序列标记为异常==



![img](https://cdn.xljsci.com/literature/174597327/page6/nog62x.png)

### 效果

<img src="https://cdn.xljsci.com/literature/174597327/page6/jh6w83.png" alt="img" style="zoom:50%;" />

![img](https://cdn.xljsci.com/literature/174597327/page6/fepdui.png)
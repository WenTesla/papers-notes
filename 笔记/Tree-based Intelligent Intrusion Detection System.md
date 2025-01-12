



## 词语本

Controller Area Network (CAN)-控制区域网络



### 数据采集

![img](https://cdn.xljsci.com/literature/125380889/page2/fnparq.png)

在CAN总线上，所有节点都是链接在一起的，当一个节点收到连接到CAN总线上的另一个设备发送的主要由ID和数据字段组成的消息时，该消息会通过IDS来检查该节点的信号线何时断开。 CAN总线从CANH变为CANL。同样，当一条消息从外网传输到内网时，也会经过网关内部的IDS并进行检查。



### 总框架

![img](https://cdn.xljsci.com/literature/125380889/page2/5002mm.png)

### 数据处理

应收集CAN消息/帧的数据，与攻击相关的主要特征是**CAN ID**和**帧的数据字段**

为了克服类不平衡数据经常导致异常检测率低的问题，SMOTE 可以生成高质量的样本，并用于所提出系统中的少数类别。 





>  过采样（Oversampling）是一种处理不平衡数据集的方法，尤其是在机器学习中。在不平衡数据集中，某些类别的样本数量远远多于其他类别，这可能导致模型偏向于多数类别，忽视少数类别。过采样的目的是通过增加少数类别的样本数量来平衡数据集，从而提高模型的性能。
>
> 常见的过采样方法包括：
>
> 1. **简单重复**：重复少数类别的样本直到达到多数类别的样本数量。
> 2. **SMOTE（Synthetic Minority Over-sampling Technique）**：生成新的合成样本，而不是简单地复制现有的少数类别样本。
> 3. **ADASYN（Adaptive Synthetic Sampling）**：根据少数类别样本的密度生成合成样本，更加适应性强。



### 机器学习的方法

所选的ML算法基于树结构，包括决策树、随机森林、额外树和XGBoost-梯度提升决策树（Gradient Boosting Decision Trees，GBDT）。



## 结果



![每一个攻击的特征重要性](https://cdn.xljsci.com/literature/125380889/page6/m276p0.png)



> 向后发送包长度标准（Bwd Packet Length Std）
>
> 转发包总长度（Total Length of FwdPackets）
>
> PUSH标志计数（PSH Flag Count）
>
> 


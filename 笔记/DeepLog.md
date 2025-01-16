# DeepLog：通过深度学习从系统日志中检测和诊断异常

## 摘要

我们提出DeepLog，一种利用长短期记忆（**LSTM**）的深度神经网络模型，将系统日志建模为自然语言序列。这允许DeepLog从正常执行中自动学习日志模式，并在日志模式偏离正常执行下从日志数据训练的模型时检测异常。此外，我们演示了如何以在线方式增量更新DeepLog模型，以便它能够随着时间的推移适应新的日志模式。此外，DeepLog从底层系统日志构建工作流，以便一旦检测到异常，用户可以诊断检测到的异常并有效地执行根本原因分析

https://netman.aiops.org/~peidan/ANM2018/9.LogAnomalyDetection/ReadingList/deeplog.pdf

## 存在的问题

- 系统或者应用越来越复杂的时候，就可能遭受更多的bug和漏洞，这些会被别人盯上，来袭击你。
- 基于标准的数据挖掘方法不再那么高效(在线的情况下，不能防御不同的攻击)
- [非结构数据](https://zhida.zhihu.com/search?content_id=131326939&content_type=Article&match_order=1&q=非结构数据&zhida_source=entity)
- 大量数据的online anamaly detection（有很多用规则）  （很多offline是针对预测已知的特殊的异常，而目标是是去检测出未知类型的log， 所以二分类是不行的）
- 有很多特征是无用的比如IP address，并且也很难知道哪些features是对于异常检测是有用的，并且不同的log类型的uuseful features都不一样
- log信息是并发的，由多线程或者并发产生的，所以导致没法用一个workflow去解决问题
- 信息有key 矩阵信息 时间，大多数模型只用到了其中一个（大多数只用key的信息）

## 结构

三个部分：

- **log key异常检测模型**
- **parameter value异常检测模型**
- **诊断检测到的异常的workflow模型。**

![img](https://cdn.xljsci.com/literature/134331373/page3/agau7i.png)

### 训练阶段

DeepLog的训练数据**是来自==正常==系统执行路径的日志条目**

![日志解析](https://cdn.xljsci.com/literature/134331373/page3/2enzbd.png)

### 检测阶段





### 威胁模型

- 导致系统执行不当行为并因此导致系统日志中异常模式的攻击。 例如Dos，BROP
- 由于系统监控服务的日志记录活动，可能在系统日志中留下痕迹的攻击。 

> Blind Return - Oriented Programming (BROP) 是一种复杂且强大的二进制漏洞利用技术。它是在传统的 Return - Oriented Programming (ROP) 基础上发展而来的，主要用于在缺乏足够的调试信息和代码可见性的情况下，对软件中的漏洞进行利用。
>
> - 首先，攻击者需要找到一个可以触发的漏洞，比如缓冲区溢出漏洞。通过向目标程序发送大量不同长度和内容的输入，攻击者观察程序的反应，例如程序是否崩溃、在什么情况下崩溃等。这就像是在黑暗中摸索，通过不断地试探来了解目标程序的一些基本特性。
> - 当程序崩溃时，攻击者可以根据崩溃信息（如返回地址的值、寄存器的值等）来推断程序的内存布局。例如，如果程序因为访问非法内存地址而崩溃，攻击者可以通过分析这个非法地址来猜测栈的位置或者其他重要的内存区域的大致位置。

## 异常检测

### 执行路径异常

![日志密钥异常检测模型概述。](https://cdn.xljsci.com/literature/134331373/page4/xotni6.png)



输入是最近日志键的历史，输出是K中n个日志键的概率分布。

相同的日志键值可能会在w中出现多次

> t是下一个要出现的日志键的序列id,

#### 训练阶段

训练阶段依赖于底层系统**正常执行**产生的一小部分日志条目。

Step1: 取设备正常运行时打印的日志，通过日志解析得到模板序列；

Step2: 按task_id（或线程号、任务号）提取模板序列；

Step3: 设置窗口长度h（通常h=10, 图中以h=3作为示意），步长s=1, 依次对每个task_id的序列进行滑动窗口提取训练样本数据。每滑动一次窗口即可得到一个训练样本，样本组合起来即可得到训练集和；

Step4: 使用训练数据和[梯度下降法](https://zhida.zhihu.com/search?content_id=165399350&content_type=Article&match_order=1&q=梯度下降法&zhida_source=entity)等算法训练神经网络。

例如，假设一个由正常执行产生的小日志文件被解析为一系列日志键：{k22，  k5，k11，k9，k11，k26}。给定窗口大小h=3，训练DeepLog的输入序列和输出标签对将是：{k22，k5，k11→k9}，{k5，k11，k9→k11}，{k11，k9，k11→k26}。

![img](https://pic4.zhimg.com/v2-747eea2d97ac30a0dde49ddb08c93fd5_1440w.jpg)

从上述收集训练数据的过程中可以发现，整个过程只要求训练数据来自于系统正常运行或故障占比很小的日志。数据标签不需要人工标注，因此该模型可以认为是一个无监督的深度学习模型。



> **类似于滑动窗口**

#### 检测阶段

DeepLog在**在线流设置中执行异常检测**。为了测试传入的日志密钥（从传入的日志条目e~t~解析）是否被视为正常或异常，我们将w={m~t−h~，…，  m~t−1~}作为其输入发送到DeepLog。**输出是一个概率分布Pr[mt|w]={k~1~：p~1~，k~2~：p~2~，…，k~n~：p~n~}，描述K中每个日志密钥在给定历史记录的情况下作为下一个日志键值出现的概率。**

在实践中，多个日志键值可能显示为mt。例如，如果系统正在尝试连接到主机，那么mt可以是“等待*响应”或“连接到*”；两者都是正常的系统行为。DeepLog必须能够在训练期间学习这样的模式。我们的策略是根据可能的日志键K的概率Pr[mt|w]对其进行排序，**如果键值属于顶级候选值，则将其视为正常值。否则，日志键将被标记为来自异常执行。**

在推理态(即检测态)，设输入模板序列**w** = [26, …, 15, 24]经过模型计算后，得到按概率大小降序排列的[条件概率分布](https://zhida.zhihu.com/search?content_id=165399350&content_type=Article&match_order=1&q=条件概率分布&zhida_source=entity) = {28:0.7595, 34:0.2103, …, 5:0.0001}。实际日志打印中存在if …  else等分支结构，例如当一个组件要与另一个组件通信，这时可能是图6中的28号模板“Waiting for * to  respond”，也可能是34号模板“Connected to  *”，这两种可能的模板都是正常的，不属于异常日志。因此比较合理的做法是取出概率值最大的*N*个模板，即topN。**如果新产生的模板在topN中，就认为对应的是正常的日志，否则认为是异常日志**。

![img](https://pic2.zhimg.com/v2-cbbbfd3b494b12fac63e8146b2b13579_1440w.jpg)

推理态步骤如下：

Step1: 取待检测的推理日志，通过日志解析得到模板序列；

Step2: 按task_id（或线程号、任务号）提取模板序列；

Step3: 加载训练后的模型，对各个task_id对应的序列滑动窗口依次检测；

Step4: 对每一个检测样本计算出概率最大topN模板集合，若则正常，否则为异常；



#### LSTM

给定一个日志键序列，训练一个LSTM网络，以最大化训练数据序列反映的下一个日志键值的概率。

![img](https://cdn.xljsci.com/literature/134331373/page4/e88aqf.png)

> 图的顶部显示了一个反映LSTM循环性质的LSTM块。每个LSTM块都将其输入的状态作为固定维度的向量记住。LSTM块的上一个时间步的状态也被输入到它的下一个输入中，连同它的（外部）数据输入（在这个特定示例中为m~t−i~），以计算新的状态和输出。这就是历史信息被传递到单个LSTM块并在其中维护的方式。
>
> 一系列LSTM块在一层中形成循环模型的展开版本，如图3中心所示。每个单元格维护一个隐藏向量H~t−i~和一个单元格状态向量C~t−i~。两者都被传递到下一个块以初始化其状态。在我们的例子中，我们为输入序列w（日志键的窗口）中的每个日志键使用一个LSTM块。因此，单层由展开的LSTM块组成

在DeepLog中，输入由日志键的窗口组成，输出是直接出现的日志键值。我们使用*分类交叉熵（categorical cross-entropy）*损失进行训练。

> 也称为 Softmax Loss。是一个 Softmax activation 加上 Cross-entropy Loss。用于multi-class classification。

### 参数值和性能异常

DeepLog通过将每**个参数值向量序列（对于日志键）视为单独的时间序列**来训练参数值异常检测模型

输入：每个时间步的输入只是该时间戳的参数值向量-The input at each time step is simply the parameter value vector from that timestamp

输出。输出是一个真实的值向量，作为下一个参数值向量的预测，基于最近历史的参数值向量序列

训练目标函数：在训练过程中使用平方损失来最小化误差。

在部署时，如果预测与观测值向量之间的误差在上述**高斯分布的高水平置信区间内**，则认为传入日志条目的参数值向量是正常的，否则认为是异常的。



### 在线更新异常检测模型

因此，DeepLog有必要在其LSTM模型中增量更新权重，以合并和适应新的日志模式

为此，**DeepLog为用户提供了一种提供反馈的机制。这允许DeepLog使用假阳性来调整其权重。**例如，假设h =  3，最近的历史序列为{k1， k2, k3}，  DeepLog以1的概率预测下一个日志键值为k1，而下一个日志键值为k2，将其标记为异常。如果用户报告这是假阳性，DeepLog可以使用以下输入输出对{k1， k2， k3→k2}来更新其模型的权重以学习这个新模式。因此，下一次给定历史序列{k1， k2，  k3}时，DeepLog可以以更新的概率输出k1和k2。



### **工作流构建-work flow construction**





## 从多任务执行构建工作流

如果我们已经排除了一个小序列是来自不同任务的共享段（即，增加训练和预测的序列长度并不会导致更确定的预测），现在的挑战是找出多键预测输出是由同一任务中的并发还是不同任务的开始引起的。我们称之为*发散点（divergence point.）*。

#### task并行的时候

![img](https://pic2.zhimg.com/v2-285f6f8b8764c21e4245029eb692c44f_1440w.jpg)

#### 有了new task

![img](https://pic1.zhimg.com/v2-6fc5d4fe3a0cf9480a0a7f35d4decb02_1440w.jpg)

#### 循环

![img](https://pic1.zhimg.com/v2-067514db0d152996eaa542c32d59b826_1440w.jpg)

### 日志条目与多个任务分离



### 使用基于密度的聚类方法

> 难以理解

### 使用工作流模型

h是历史序列窗口

g是预测的窗口值



## 评估

### 数据集

- HDFS
- OpenStack

### 指标

![对HDFS日志进行评估。](https://cdn.xljsci.com/literature/134331373/page10/ykw0it.png)

![OpenStack日志评估。](https://cdn.xljsci.com/literature/134331373/page10/446byd.png)



![Blue Gene/L日志的评价](https://cdn.xljsci.com/literature/134331373/page11/3sa73i.png)

## 总结

本文介绍了DeepLog，这是一个使用基于深度神经网络的方法进行在线日志异常检测和诊断的通用框架。DeepLog学习和编码整个日志消息，包括时间戳、日志键和参数值。它在每个日志入口级别执行异常检测，而不是像许多以前的方法那样仅限于每个会话级别。DeepLog可以从日志文件中分离出不同的任务，并使用深度学习（LSTM）和经典挖掘（密度聚类）方法为每个任务构建工作流模型。这可以实现有效的异常诊断。通过结合用户反馈，DeepLog支持对其LSTM模型进行在线更新/培训，因此能够整合和适应新的执行模式。 对大型系统日志的广泛评估清楚地证明了DeepLog与以前方法相比的卓越有效性。未来的工作包括但不限于将其他类型的RNN（递归神经网络）纳入DeepLog以测试其效率，以及整合来自不同应用程序和系统的日志数据以执行更全面的系统诊断（例如，MySQL数据库的故障可能是由单独系统日志中反映的磁盘故障引起的）。

## 不足

**DeepLog输入数据的编码方式为[one-hot](https://zhida.zhihu.com/search?content_id=165399350&content_type=Article&match_order=4&q=one-hot&zhida_source=entity)，所以无法学习出两个模板之间的语义相似度**，例如，假如[模板数据库](https://zhida.zhihu.com/search?content_id=165399350&content_type=Article&match_order=1&q=模板数据库&zhida_source=entity)的表中共有3个模板，如表1所示。从模板ID或者[one-hot编码](https://zhida.zhihu.com/search?content_id=165399350&content_type=Article&match_order=3&q=one-hot编码&zhida_source=entity)无法学习出1号模板与2号模板业务意义相反，也学不到1号模板与3号模板业务意义相近。因此，原始的DeepLog的学习能力是有局限性的。

| 模板ID | 模板内容                               | One-hot编码 |
| ------ | -------------------------------------- | ----------- |
| 1      | Interface * changed state to down      | [1, 0, 0]   |
| 2      | Vlan-Interface * changed state to up   | [0, 1, 0]   |
| 3      | Vlan-Interface * changed state to down | [0, 0, 1]   |





## 参考文献

https://zhuanlan.zhihu.com/p/194371740

https://zhuanlan.zhihu.com/p/347712013

https://www.youtube.com/watch?v=At19CBGpbMI


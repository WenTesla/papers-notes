# Adversarial attack detection framework based on optimized weighted conditional stepwise adversarial network基于优化加权条件逐步对抗网络的对抗攻击检测框架

## 摘要

本研究提出了一个 WCSAN-PSO 框架，用于检测 IDS 中的对抗性攻击，该框架基于加权条件逐步对抗网络 (WCSAN)，并采用粒子群优化 (PSO) 算法和 SVC（支持向量分类器）进行分类



> 入侵检测的主要目标是区分正常和异常信息泄露[^1]。
>
> IDS 可分为基于主机的 (HIDS) 和基于网络的 (NIDS) 方法

## 本文的贡献

- 提出用于对抗性攻击中的入侵检测的 **WCSAN-PSO** 框架。 
- 通过结合特征提取（主成分分析）和特征提取（最小绝对收缩和选择算子）来分析框架
- 利用标签攻击来识别使用签名的已知攻击。可以在初始级别进行预测，从而减少 IDS 中的带宽、计算资源和攻击检测效率。
- 根据 IDS 流量特征生成对抗样本。 IDS 使用训练数据集进行训练，包括从 WCSAN 获取的真实网络流量样本和攻击网络流量样本。
- 使用优化的 PSO 算法和 SVC 分类器以及 CIC-IDS2017 数据集（包含 IDS 中不同类型的当代攻击）来开发和评估该框架。

为了克服现有的研究差距，所提出的框架设计了独特的攻击分级模式，同时维护和更新签名数据库，以便检测到任何已知的攻击。可以在初始级别进行预测，从而减少 IDS 中的带宽、计算资源和攻击检测效率。所提出的框架利用 WCSAN 构建具有正确标签的校正训练数据集。  PCA采用了特征提取和LASSO进行特征选择。 PSO算法优化了WCSAN中生成器和判别器的参数，以增强IDS的对抗训练。

> Weighted conditional stepwise adversarial network-WCSAN-加权条件逐步对抗网络
>
> 



## 介绍

> IDS分类：HISD与NIDS
>
> HIDS 安装在单个信息主机上。任务是监视单个主机上的所有活动，扫描安全策略违规和可疑活动[29]。
>
> NIDS安装在网络上，以保护所有设备和整个网络免受入侵。 NIDS 持续观察网络流量以检测安全漏洞和违规行为 [30]

根据所使用的模型，IDS 可以分为两类：基于签名的 IDS 和基于异常的 IDS。

基于签名的 IDS 将预定义的攻击签名存储在数据库中，并监视网络以查找与这些签名的任何匹配项。基于异常的 IDS 监视网络流量并将其与网络的标准使用模式进行比较 [31]

![IDS 中的对抗性攻击](https://cdn.xljsci.com/literature/126751669/page3/n3d2hh.png)

生成对抗网络（GAN）是一类有效的生成模型，它采用在零和游戏中同时训练的两个网络，一个网络专用于数据生成，另一个网络专用于区分[32]。GAN 由两个元素组成：生成器和判别器。生**成器模拟数据分布以创建对抗性示例并欺骗鉴别器，鉴别器试图区分假示例和真实示例[33]。**

### 他人的成果

![img](https://oss.xljsci.com//literature/126751669/page0/1733751836275.png)

![img](https://oss.xljsci.com//literature/126751669/page0/1733751869685.png)

### 问题阐述

![img](https://cdn.xljsci.com/literature/126751669/page7/lakej2.png)

## 数据采集

https://www.unb.ca/cic/datasets/ids-2017.html-公开数据集

如下图

![img](https://oss.xljsci.com//literature/126751669/page0/Screenshot 2024-12-10 at 10-01-59 Index of _CICDataset_CIC-IDS-2017_Dataset_CIC-IDS-2017_CSVs1733796125343.png)



## 使用特征提取进行主成分分析

采用PCA，m维网络流量变量可以减少l维网络流量特征[61]。

PCA步骤如下

分组
$$
\alpha=\frac1j\sum_{n=1}^jh_a
$$
使用样本均值，使用等式计算测试集的协方差矩阵
$$
P=\frac1j\sum_{a=1}^j(h_a-\alpha)(h_a-\alpha)^o
$$
P是样本集的相关矩阵。

可以使用等式来识别样本协方差矩阵的特征值和向量
$$
P=K.\sum.K \\
\sum=diag(\lambda_1,\lambda_2,....\lambda_s) \lambda_1>=\lambda_2>=,....\lambda_s>=0 \\
K=[k_1,k_2,k_3]\\
$$
P是m个已对角组织并降序排列的协方差矩阵的质量值；下面显示了协方差矩阵 λj 的属性值以及属性向量。

对于前 l 行主要项目，使用从前l行主要成分产生的特征向量和特征评级来计算累积偏差养老金缴款。
$$
\theta=\frac{\sum_{j=1}^l\lambda_j}{\sum_{i=1}^n\lambda_i}
$$
使用公式12和13利用和减少收集到的具有q行特征的向量大小。
$$
A=K_l\\
X=A.Y \\
$$

## 基于lasso的特征选择与标记攻击检测

它减少了未利用的平方和，被迫提交整个相关系数估计的总和小于完全符合。

首先，收集有关网络流量行为的信息用于系统分析。数据收集后，信息被标记，以区分已知和未知的行为。当系统检测到一个与已知签名不同的攻击时，系统使用建议的框架来识别和分类未知或新颖的攻击。

如下图

![img](https://cdn.xljsci.com/literature/126751669/page10/og4b63.png)

## 处理类不平衡问题

SMOTE

## ==WCSAN==

![Architecture of the WCSAN-based IDS](https://cdn.xljsci.com/literature/126751669/page10/47nr4x.png)

![img](https://oss.xljsci.com//literature/126751669/page0/1733812250469.png)

<img src="https://cdn.xljsci.com/literature/126751669/page14/oupd90.png" alt="流程图" style="zoom:67%;" />

## 场景

总结如下

![Three evaluation scenarios for the analysis of IDS in adversarial attacks](https://cdn.xljsci.com/literature/126751669/page15/z5o7g6.png)

### 场景1

对原始网络流量数据集进行预处理和归一化，使用PCA提取特征，并使用LASCO选择特征。

Table 6 The IDS before adversarial attacks on the dataset

![Table 6 The IDS before adversarial attacks on the dataset](https://cdn.xljsci.com/literature/126751669/page15/gbdrss.png)

### 场景2

对抗性样本是使用 WCSAN 生成的，没有防御机制，也没有经过对抗性训练。场景 2 中使用不平衡数据集，并使用组合对抗样本和原始训练数据集转换提取的特征，如表 7 所示。结果用测试数据集进行测试。然**而，在有对抗样本的网络流量数据集上测试的无防御机制的 IDS 的准确率、精确度、召回率和 F1 分数低于没有对抗样本的 IDS。**

![img](https://cdn.xljsci.com/literature/126751669/page16/eohh8f.png)

### 场景3

IDS 在**组合数据集（即正常原始流量和场景 2 生成的对抗样本）**上进行进一步训练。**IDS 使用使用建议的 WCSAN-PSO 防御生成的校正对抗训练数据集进行训练**。

这表明与场景 2 相比，在应用防御机制后，所提出的框架提高了对抗场景中检测恶意攻击的性能。



![Table 13 Performance analysis of the proposed framework in the imbalanced dataset](https://cdn.xljsci.com/literature/126751669/page19/cabk6a.png)

![img](https://oss.xljsci.com//literature/126751669/page0/1733816767403.png)

## 总结

本研究提出了一种基于 WCSAN-PSO 的框架，采用加权条件逐步对抗网络，采用粒子群优化和支持向量分类器进行分类，以有效检测 IDS  中的对抗攻击。该框架在第一阶段使用更新的基于签名的攻击检测来预测已知的攻击，从而减少了计算资源。该研究通过三个综合场景分析了对抗性攻击和防御机制，并进行了实际和定量的评估。所提出的框架在确定正常流量方面的准确率达到了 99.36%，在对抗性攻击场景中识别恶意流量的准确率达到了 98.55%。所提出的框架在平衡数据集中产生的 AUC 值为  0.99，在使用不平衡数据集时产生的 AUC 值为  0.97，这表明了一致性。攻击者可能会修改许多网络流量特征而不影响网络行为，从而难以检测入侵。未来的目标是研究所提出的框架对各种机器学习和深度学习技术的影响。这种方法可以扩展到利用先进技术探索对抗性机器学习中的可迁移性概念
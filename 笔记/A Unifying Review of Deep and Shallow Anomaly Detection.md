## A Unifying Review of Deep and Shallow Anomaly Detection-深度和浅层异常检测的统一综述

本文聚焦异常检测（AD）领域，旨在建立传统 “浅层” 异常检测方法与新兴深度学习方法的关联，揭示二者间的交叉融合潜力，同时从统一视角梳理现有方法的共性原理与隐含假设，为该领域提供系统性框架，并通过实证评估与实例分析给出实践建议，最后指出关键开放挑战与未来研究方向。



| 缩写  | 全称                                  | 中文                 |
| ----- | ------------------------------------- | -------------------- |
| AD    | Anomaly detection                     | 异常检测             |
| AE    | Autoencoder                           | 自动编码器           |
| AP    | Average precision                     | 平局精度             |
| AAE   | Adversarial Autoencoder               | 对抗自动编码器       |
| AUPRC | Area under the precision–recall curve | 精确率               |
| AUROC | Area under the ROC curve              |                      |
| CAE   | Contrastive Autoencoder               | 对比自动编码器       |
| DAE   | Denoising Autoencoder                 |                          |
| DGM   | Deep generative model                 |  |
| DSVDD | Deep support vector data description  | 深度支持向量机数据描述 |
| DSAD | Deep semisupervised AD | 深度半监督异常检测 |
| ELBO      | Evidence lower bound                          | 证据下界                 |
| GAN       | Generative adversarial network                | 生成对抗网络             |
| GMM       | Gaussian mixture model                        | 高斯混合模型             |
| iForest   | Isolation forest                              | 孤立森林                 |
| GT        | Geometric transformation                      | 几何变换                 |
| KDE       | Kernel density estimation                     | 核密度估计               |
| k - NN    | k - nearest neighbors                         | k 近邻                   |
| kPCA      | Kernel principal component analysis           | 核主成分分析             |
| LOF       | Local outlier factor                          | 局部离群因子             |
| LPUE      | Learning from positive and unlabeled examples | 从正例和无标签示例中学习 |
| LSTM      | Long short - term memory                      | 长短期记忆               |
| MCMC      | Markov chain Monte Carlo                      | 马尔可夫链蒙特卡罗       |
| MCD       | Minimum covariance determinant                | 最小协方差行列式         |
| MVE       | Minimum volume ellipsoid                      | 最小体积椭球             |
| OOD       | Out - of - distribution                       | 分布外                   |
| OE        | Outlier exposure                              | 异常暴露                 |
| OC - NN   | One - class neural network                    | 单类神经网络             |
| OC - SVM  | One - class support vector machine            | 单类支持向量机           |
| pPCA      | Probabilistic principal component analysis    | 概率主成分分析           |
| PCA       | Principal component analysis                  | 主成分分析               |
| PDF       | Probability density function                  | 概率密度函数             |
| PSD       | Positive semidefinite                         | 半正定                   |
| RBF       | Radial basis function                         | 径向基函数               |
| RKHS      | Reproducing kernel Hilbert space              | 再生核希尔伯特空间       |
| rPCA      | Robust PCA                                    | 鲁棒主成分分析           |
| SGD       | Stochastic gradient descent                   | 随机梯度下降             |
| SGLD      | Stochastic gradient Langevin dynamics         | 随机梯度朗之万动力学     |
| Semi - AD | Semisupervised AD                             | 半监督异常检测           |
| VAE       | Variational AE                                | 变分自动编码器           |
| SVDD      | Support vector data description               | 支持向量数据描述         |
| VQ        | Vector quantization                           | 向量量化                 |
| XAI       | Explainable AI                                | 可解释人工智能           |

![img](https://cdn.xljsci.com/literature/181566286/page3/zcq6ek.png)

- **Classification（分类类）**：把异常检测视为分类问题，区分正常样本和异常样本。
  - 浅层方法：如支持向量数据描述（SVDD）、单类支持向量机（OC - SVM）等，通过构建决策边界来划分正常与异常区域。
  - 深层方法：如基于深度神经网络的单类分类模型（OC - NN）、深度支持向量数据描述（DSVDD）等，利用深度网络学习更复杂的分类边界。

- **Probabilistic（概率类）**：基于**概率**模型，通过估计数据的概率分布来识别异常（概率低的样本为异常）。
  - 浅层方法：如直方图（Histogram）、核密度估计（KDE）、高斯混合模型（GMM）等，是传统的概率密度估计方法。
  - 深层方法：如能量基模型（EBMs）、变分自编码器（VAEs）、归一化流（Flows）、生成对抗网络（GAN，包括判别器 GAN(D) 和生成器 GAN相关应用）等，利用深度模型进行更复杂的概率分布建模。

- **Reconstruction（重建类）**：通过学习正常数据的重建模型，异常样本因难以被准确重建而被检测出来（重建误差大的为异常）。
  - 浅层方法：如主成分分析（PCA）、核主成分分析（kPCA）、向量量化（VQ）、k - 均值（k - Means）等，基于线性或传统非线性方法进行数据重建。
  - 深层方法：如对抗自编码器（AAEs）、卷积自编码器（CAEs）、去噪自编码器（DAEs）等，利用深度自编码器学习更精准的重建表示。
- **Distance（距离类）**：基于样本间的距离或密度差异来检测异常，距离远或密度异常的为异常样本。包括 k - 最近邻（k - NN）、局部离群因子（LOF）、孤立森林（iForest）等方法，这些方法相对更 “浅层”，直接基于样本在原始或简单特征空间的距离关系进行异常判断。

PLE使用的是概率类加上距离类的方法



## 基本概念

### 1. 异常定义与分类

- **定义**：异常是显著偏离 “正常概念” 的观测值，从概率角度可表述为：设正常数据分布为*p*+，异常集A={*x*∈X∣p^+^(x)≤*τ*}（*τ*为阈值，且P^+^(A)足够小）。

- **分类**：
  
  - **点异常**：单个异常数据点（如欺诈交易、受损产品图像）。
  - **条件 / 上下文异常**：特定上下文下的异常（如亚马逊雨林低于冰点的日均温、1997 年后 1 美元的苹果股票价格）。
  - **群体 / 集体异常**：相关数据点构成的异常集合（如网络攻击集群、异常时间序列片段）。
  - **低阶感官异常与高阶语义异常**：前者涉及低层次特征（如图像纹理缺陷、文本字符错误），后者涉及高层语义（如非正常类别的物体图像、语义错误的文本）。
  
  <img src="https://cdn.xljsci.com/literature/181566286/page5/gncqbp.png" alt="img" style="zoom: 67%;" />
  
- **术语辨析**：异常（来自与正常分布不同的分布）、离群值（正常分布中低概率实例）、新样本（非平稳正常分布的新区域实例），本文统一称为 “异常”。

### 2. 核心假设与问题形式化

- **集中假设**：正常数据的高概率区域可被界定，即存在阈值\(\tau\)，使正常区域\(\mathcal{X} \setminus \mathcal{A}=\{x \in \mathcal{X} | p^+(x)>\tau\}\)非空且体积小（勒贝格测度意义上）。
- **密度水平集估计**：AD 目标等价于估计正常分布的低密度区域，\(\alpha\) - 密度水平集\(C_{\alpha}\)是概率至少为\(1-\alpha\)的最小密度水平集，可通过密度估计、阈值设定等方式实现，且能基于此定义阈值异常检测器\(c_{\alpha}(x)\)。

### 3. 数据集设置与数据属性

- **数据集设置**：
  - 无监督设置：**仅含无标签数据**，需考虑数据中的噪声（如测量不确定性）与污染（如未检测到的异常混入，数据分布为\(\mathbb{P} \equiv (1-\eta)\mathbb{P}^+ + \eta\mathbb{P}^-\)，\(\eta\)为污染率）。往往使用unsupervised model，这时候假设**正常样本是聚集的，异常样本是分散的。**由于正常样本占了所有样本的绝大部分，模型将所有样本看作正常样本进行建模，学习到正常样本的pattern，那么就可以将与正常pattern偏离、分散的异常样本检测出来。
  - 半监督设置：含无标签与少量有标签数据（正常\(\bar{y}=+1\)、异常\(\bar{y}=-1\)），常见 “正例与无标签示例学习（LPUE）” 场景，还可利用辅助异常数据（如 OE 方法）。**如果异常样本也有一定的模式，而不是完全无规律的散布在整个样本空间**，那么我们有理由相信用部分带标签的异常样本能够显著提升模型的检测性能。
  - 监督设置：含完全标签数据，但因异常标签难具代表性，更应视为 “标签知情的密度水平集估计”，实际中较少见。
- **数据属性**：包括规模（样本量、标签量）、维度（低 / 高维）、类型（连续 / 离散 / 分类）、模态（单 / 多模态）、凸性（数据支撑集凸 / 非凸）、相关性（特征线性 / 非线性相关）等，需在建模中体现对应假设。

## 异常检测算法

### Density Estimation and Probabilistic Models

这一类的模型思想可能是最好理解的：**对于正常样本进行[density estimation](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=density+estimation&zhida_source=entity)，从而学习到正常样本的分布，并把在落在这个分布外的样本视为异常。**例如考虑单个维度的数据，用均值为 ，[标准差](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=标准差&zhida_source=entity)为 的[正态分布](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=正态分布&zhida_source=entity)对正常样本进行描述，并将数据落在 之外的样本视为异常样本。

- Classic Density Estimation: 例如用[多元高斯分布](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=多元高斯分布&zhida_source=entity)拟合训练数据，并通过对数似然值的大小判断测试样本是否为异常。
- Energy-based Models：在前几年比较流行的算法，像deep belief networks以及deep Boltzmann machines都属于energy-based models。模型通过energy function来刻画density。
- Neural Generative Models (e.g., VAEs and GANs)：训练模型学习一个[映射关系](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=映射关系&zhida_source=entity)——将从predefined distribution（e.g., normal distribution or uniform distribution）中采样的[vector](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=vector&zhida_source=entity)映射至样本的真实分布。
- Normalizing Flows：normalizing flow的方法学习到的[latent vector](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=latent+vector&zhida_source=entity)与输入是同维度的，并且网络中权重矩阵被设计成可逆的，以使得整个网路是可逆的。



### One-Class Classification

**One-class classification在于用一个映射将正常样本投影到一个高维空间中，并将投影至该[高维空间](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=2&q=高维空间&zhida_source=entity)之外的样本视为异常样本**。这个映射可以是预先定义的（例如kernel-based方法），也可以通过神经网络进行学习。

- Kernel-based One-class classification：典型的有OC-SVM以及SVDD算法，前者利用[超平面](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=超平面&zhida_source=entity)将输入映射特征与其初始点的间隔尽可能最大化，而后者则在[特征空间](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=特征空间&zhida_source=entity)中利用超球体将训练数据点尽可能囊括，并使得该超球体的体积尽可能的小。
- Deep One-class classification：与kernel-based 方法不同，deep learning无需定义一个[先验的](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=先验的&zhida_source=entity)映射函数（kernel），而是让网络自己学习这个映射关系。典型的算法有Deep SVDD方法，将上述SVDD的思想用神经网络进行训练，其中loss function如下所述， 为相应的正则化项。





### Reconstruction Models

Reconstruction-based model也比较好理解，**模型的目标在于学习正常样本的pattern——如何将正常样本进行有效重构**（例如将数字1重构为一个颜色、角度、光泽都接近的数字1，都并不完全一样）。在学习了关于正常样本的pattern之后，**模型对于正常样本能够成功重构，但对于异常样本往往重构失败**，因此将[重构损失](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=重构损失&zhida_source=entity)视作异常得分就能很好的将异常检测出来。

- PCA：PCA不仅可以拿来降维，也可以视作对于样本在[线性空间](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=线性空间&zhida_source=entity)中的一种重构
- Autoencoder：神经网络版本的PCA。如果不在auto-encoder中加入[非线性激活层](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=非线性激活层&zhida_source=entity)，那么auto-encoder则等价于PCA

![img](https://cdn.xljsci.com/literature/181566286/page10/yssn6j.png)



## 使用标签

- artifical labeled data——**获取不到带标签的异常样本，那我可以造样本**：）！最简单的思路是从假定服从[uniform distribution](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=2&q=uniform+distribution&zhida_source=entity)的分布中抽取异常样本，让异常样本扩张到整个样本空间，这样就将anomaly detection问题有效转化成了[binary classification](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=binary+classification&zhida_source=entity)问题。不过这个形式在高维空间往往不可行，因为高维空间中抽取的样本数量与[特征维数](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=特征维数&zhida_source=entity)呈现指数级的对应关系。**另外一个比较好的思路是用生成式模型造样本，例如用GAN。**
- auxilary labeled data——第二个方法是**利用较容易获得的公开数据集作为异常样本**，例如在CV中利用摄影网站的一些照片集，NLP中利用维基百科的语料。实际上此类方法有专有的术语，称为[Outlier Exposure (OE)](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1812.04606.pdf)，发表在ICLR 2019上。
- true labeled data——**获取大量的标注样本往往是困难的，那我们可不可以标注非常少量的数据呢？**很多学术论文证明，即便可获得的标注数据量很少，也能够显著提升算法的检测性能。例如ICLR 2020中Deep SAD就在Deep SVDD的基础上做出了改进（Deep SVDD的loss function请阅读上文）——思想也很直观，对于正常样本以及无标签样本的训练方式不变，**使其特征[空间映射](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=空间映射&zhida_source=entity)尽可能地离超球面中心近**；而对于异常样本而言，令其计算距离的一项以*倒数*的形式计算（异常样本的标签可以设为-1），**使得异常样本的特征映射尽可能离[超球面](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=2&q=超球面&zhida_source=entity)的中心远。**



## 展望

### 异常检测领域可结合的方向：

- anomaly detection algorithms for noise and contamination data

- Bayesian inference for deep learning

- using labeled data information in probabilistic model and reconstruction-based model

- active learning in AD

### 异常检测领域待需改进的问题：

- **robustness**：模型的[鲁棒性](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=2&q=鲁棒性&zhida_source=entity)包括考虑模型识别异常时的置信度；考虑对训练样本分布外的异常（out-of-distribution anomalies）进行识别；open set recognition；对于adversarical examples and attacks的鲁棒性；
- **可解释的[异常检测模型](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=异常检测模型&zhida_source=entity)**：如何将已在supervised task中广泛应用的解释性方法迁移至异常检测任务中；
- **更多的公开数据集**：目前很多学术文章中的[dataset](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=dataset&zhida_source=entity)并不是真正意义上的异常检测数据集，例如很多学术文章中将MNIST、FashionMNIST以及CIFAR10中一个class视为anomaly，剩余的class视为normal，这种做法可能并不能反映出模型的真实检测性能。**结合上文章之前的结论——没有一个模型能够“包打天下”，这些在论文中声称比较好的模型可能也只是在部分数据集上表现好，在其余数据集中表现会显著下降。**
- **weak supervision and self-supervised model**：获得大量的标注数据成本较高，能否通过模型有效利用weak supervision下低成本的标注数据？（根据[周志华](https://zhida.zhihu.com/search?content_id=189903403&content_type=Article&match_order=1&q=周志华&zhida_source=entity)老师的[paper](https://link.zhihu.com/?target=https%3A//watermark.silverchair.com/nwx106.pdf%3Ftoken%3DAQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAArswggK3BgkqhkiG9w0BBwagggKoMIICpAIBADCCAp0GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMPVGAJY2DS9MVtx_rAgEQgIICbiBRatDXcGHvdnferOb_6uxyr1uyX2GzyTbzmRw-u23usUJGWZ8_YhihkADijJHIiosPqBSIxetaNla2kf_ubyf7V34GIjuMbbk8fF2mUoHeUc1su6yqwLQnebKtNDDo5iTjWZHYkKIPoxCEsjd5C_T_utKcx01uQ6NBaxySSM91AmKtFSwymrugJ_63DpAnVTp-ee2zRJK2kgDBVl7dcwVnSw2mFNz8cu5RR5L5Iw45wj8l8zQTqNMrZOkRd_gwOdMlR738PR-5ZFVMMh3HfMfbLMMYY1l3nU85pkDkw9t6ksTL5sKlJgaMIdK0i9Sq2Rwlr_LwvNyxZXHPfn708cJcgIWxs-naUzP-t8Hht_xEQPEaQvkEoJ0DL29FV2yFl7fKh8Qt47LQbTIEPUkP5eX-r-E8KvJ-sCqQaueFY5WL5T_DlukAgn0-l2rS8xqcfhyLswH-4cEmB05Ew8Pc1lTHT36-bKW8ETPrf8vCFoES37t8fz7ycaJ486EWZHK3K94aal51ardYYTdxIDYUT86EjXMufsWOcQO0YWVbmu7E0CbmpsZVLSXupbv7QXAyZ-t26jRfHecPgz1bAWDJ2hARTYnoK5_llZQj99wc8dt0yoXUZGnbbNn0ltniDO-CtI3AWTfg9gRJF0uiTU_2Vu-vLePL6w3SWSXYsdv_Q3TXkkIudQb1oE--15QTEL6t9We-D0nhyzrvPrnZ0Az5Uh8sXdTDwe-sqcJOWwdcEf_BBZjihjCZIkSCpvoc-koVMK95Wzu_EnfaJhg2YdZYTmpfDVsHxtbNk7wCA0zLnWC8MIgpNf0Xw11WI-4kifQ)，这些标注样本可能是incomplete, inexact或者inaccurate的）；或者将已在CV和NLP领域大放异彩的self-supervised方法用在anomaly detection task中？
- **考虑模型本身的设定与异常检测问题的矛盾**：例如考虑用auto-encoder对手写数字“1”进行学习并有效重构。成功训练的模型能够对手写数字“8”重构失败，并将其检测出异常，因为数字“8”与“1”差别比较明显。但输入一张背景全为黑色的图片，模型往往不能检测出这是异常，因为数字“1”对应的样本中本身包含大量黑色背景。**这个结果对于auto-encoder设定而言是没错的，黑色背景也为其学习到的内容，所以应该重构成功，但对于异常检测问题而言这是我们不希望看到的。如何处理这种矛盾？**







## 参考文献

 [用统一视角看待异常检测中的shallow model与deep learning model - 知乎](https://zhuanlan.zhihu.com/p/457976793)

[高斯分布 | Stand Alone Complex](https://blog.puqing.work/blog/971deee9)

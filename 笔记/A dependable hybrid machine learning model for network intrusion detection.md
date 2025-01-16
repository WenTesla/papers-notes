# A dependable hybrid machine learning model for network intrusion detection

## 摘要

网络入侵检测系统（NIDS）在计算机网络安全中发挥着重要作用。在多种检测机制中，基于异常的自动检测明显优于其他机制。随着攻击的复杂性和数量的不断增加，处理大量数据是基于异常的 NIDS  开发中一个公认的问题。然而，当前的模型在所需的准确性和可靠性方面是否满足当今网络的需求？在这项研究中，我们提出了一种新的混合模型，该模型结合了机器学习和深度学习，以提高检测率，同时确保可靠性。我们提出的方法通过结合用于**数据平衡的 SMOTE 和用于特征选择的 XGBoost  来确保高效的预处理**。我们将我们开发的方法与各种机器学习和深度学习算法进行比较，以便找到更有效的算法来在管道中实现。此外，我们根据一组基准性能分析标准选择了最有效的网络入侵模型。我们的方法在**两个数据集 KDDCUP'99 和 CIC-MalMem-2022** 上进行测试时产生了出色的结果，KDDCUP'99 和 CIC-MalMem-2022  的准确度分别为 99.99% 和 100%，并且没有过度拟合或 Type-1和 2 类问题。



## 贡献

我们提出了一种新颖的混合方法，并通过解释模型的准确性、可用性和可扩展性指标的可靠性来展示其检测网络入侵的可靠性。

用于特征选择的 XGBoost、用于数据平衡的 SMOTE 以及适当的预处理，我们开发了自己的混合方法，该方法独特地优于最先进的网络入侵检测模型。

## 特征选择

![特征选择](https://cdn.xljsci.com/literature/128113458/page7/xxaybu.png)

一步步减少，直到达到最好的效果。

![img](https://cdn.xljsci.com/literature/128113458/page7/rks50m.png)

## 框架

![img](https://cdn.xljsci.com/literature/128113458/page8/927ohx.png)

![img](https://cdn.xljsci.com/literature/128113458/page11/sde2dh.png)

## 结果

![img](https://cdn.xljsci.com/literature/128113458/page13/42ln1a.png)

![Table 9 Performance analysis for binary classification.](https://cdn.xljsci.com/literature/128113458/page13/zvqrti.png)

![Fig. 10. Performance analysis graphs for binary classification.](https://cdn.xljsci.com/literature/128113458/page13/37q0ag.png)

![Fig. 11. Confusion matrix for binary classification of KDDCUP’99.](https://cdn.xljsci.com/literature/128113458/page14/axv4f6.png)

![Fig. 13. Performance comparison graphs for multilabel classification.](https://cdn.xljsci.com/literature/128113458/page15/9dofa1.png)

![Fig. 14. Performance analysis graphs for multilabel classification.](https://cdn.xljsci.com/literature/128113458/page16/n78ppe.png)

![Fig. 15. Confusion matrix for multilabel classification of KDDCUP’99.](https://cdn.xljsci.com/literature/128113458/page17/dzorev.png)

![Fig. 17. Performance comparison graphs for binary classification.](https://cdn.xljsci.com/literature/128113458/page18/y39xyj.png)

![Fig. 18. Performance analysis graphs for binary classification.](https://cdn.xljsci.com/literature/128113458/page19/jcinoa.png)

![Fig. 19. Confusion matrix for multilabel classification of CIC-MalMem-2022.](https://cdn.xljsci.com/literature/128113458/page20/6du7hf.png)
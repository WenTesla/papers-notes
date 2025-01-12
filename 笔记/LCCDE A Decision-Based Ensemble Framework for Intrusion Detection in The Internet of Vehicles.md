# LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles

### 专业词语



Leader Class and Confidence Decision Ensemble (LCCDE)

Exclusive Feature Bundling (EFB)

Gradient-based One-Side Sampling (GOSS)



### 摘要

为了准确检测车联网网络中的各种类型的攻击，我们提出了一种新颖的集成 IDS 框架，名为领导类和置信决策集成 (LCCDE)。它是通过针对每种类别或类型的攻击确定三种高级 ML  算法（XGBoost、LightGBM 和 CatBoost）中性能最佳的 ML  模型而构建的。然后，利用类别领导者模型及其预测置信度值来做出有关检测各种类型网络攻击的准确决策。在两个公共 IoV  安全数据集（Car-Hacking 和 CICIDS2017 数据集）上的实验证明了所提出的 LCCDE 对于车内和外部网络入侵检测的有效性。



### 贡献

- 它提出了一种名为 LCCDE 的新型集成框架，可使用类领导者和置信度决策策略以及梯度提升 ML 算法在车联网中进行有效的入侵检测。
- 它使用两个公共 IoV 安全数据集 Car-Hacking [13] 和 CICIDS2017 [14] 数据集（分别代表 IVN 和外部网络数据）来评估所提出的框架。
- 它将所提出的模型的性能与其他最先进的方法进行了比较

### 框架

两个阶段：模型训练和模型预测。在模型训练阶段，在车联网流量数据集上训练三种先进的机器学习算法：XGBoost [10]、LightGBM  [11] 和 CatBoost  [12]，以获得所有类别/类型攻击的领导者模型。在模型预测阶段，使用类领导者模型及其预测置信度来准确检测攻击。本节提供了算法的详细信息。



![img](https://cdn.xljsci.com/literature/125823255/page3/osucj3.png)



算法

![img](https://oss.xljsci.com//literature/125823255/page0/1733195522418.png)
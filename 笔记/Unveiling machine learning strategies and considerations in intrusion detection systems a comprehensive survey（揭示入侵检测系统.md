# Unveiling machine learning strategies and considerations in intrusion detection systems: a comprehensive survey（揭示入侵检测系统中的机器学习策略和注意事项：全面调查）

## 摘要

因此，组织正在 IDS 中使用机器学习 (ML) 和深度学习 (DL) 算法来实现更准确的攻击检测。本文概述了  IDS，包括其类和方法、检测到的攻击以及所使用的数据集、指标和性能指标。对最近关于基于 IDS  的解决方案的出版物进行了彻底的审查，评估了它们的优点和缺点，并讨论了它们的潜在影响、研究挑战和新趋势。

## IDS



![术语表](https://cdn.xljsci.com/literature/127197045/page3/18hjcl.png)

![img](https://oss.xljsci.com//literature/127197045/page0/1733970288477.png)

## 基于检测方法的IDS

### 误用检测

基于签名或滥用检测方法基于作为模式或规则存储在系统中的已知签名。将每个收到的数据包与提供的签名进行比较（Lansky 等人，2021）。当有匹配时，计划会发送警报。滥用检测可以有效检测频繁的网络攻击，**但无法检测新的网络攻击。**

### 异常检测

异常检测方法基于创建配置文件来区分正常行为和攻击行为。使用多个提取或生成的特征来检查每个传入数据包，**以确定它是正常的还是恶意的**（Kunhare 和 Tiwari，2018）。当检测到攻击活动时，会发出警报。**与误用检测方法相比，异常检测方法可以有效地检测新的攻击，但代价是较高的 FAR**（false alarm rate）。

### 混合检测

混合 IDS 结合了异常检测和误用检测方法，比任何一种方法都更有效

## 入侵检测系统检测到的重大网络攻击

![img](https://oss.xljsci.com//literature/127197045/page0/1733970824475.png)

## AI

![img](https://oss.xljsci.com//literature/127197045/page0/1733971612486.png)

![img](https://oss.xljsci.com//literature/127197045/page0/1733971688052.png)



## 归纳

![img](https://cdn.xljsci.com/literature/127197045/page11/7gp763.png)

![数据集](https://cdn.xljsci.com/literature/127197045/page12/bgfrk7.png)



![基于机器学习和深度学习的 IDS 方法的优点和缺点](https://cdn.xljsci.com/literature/127197045/page14/bp72tq.png)

![img](https://cdn.xljsci.com/literature/127197045/page15/92sj2m.png)

## 基于机器学习和深度学习的入侵检测系统的研究挑战

因此，IDS 研究的挑战之一是系统地开发一个当代数据集，其中包含几乎所有攻击类型的足够示例。

因此，需要一种可行的技术来选择最重要的属性，同时最小化计算和处理开销。 






























# Revisiting Automotive Attack Surfaces: a Practitioners’ Perspective(重新审视汽车攻击面：从业者的视角)

## 缩写

- **IVN(In vehicle network，车载网络，)**
- **TARA（Threat Analysis and Risk Assessment，威胁分析和风险评估；威胁分析与风险评估，是汽车电子电气架构中常用的[网络安全](https://zhida.zhihu.com/search?content_id=219454626&content_type=Article&match_order=1&q=网络安全&zhida_source=entity)威胁分析与风险评估方法论。TARA从道路交通参与者角度，确定道路交通参与者受威胁场景影响的程度）**
- ISO21434 （一项关于汽车网络安全的国际标准）
- ECU（electronic control units）

## Abstract

> we built an improved threat database for automotive systems using the collected interview data, which enhanced the existing database both quantitatively and qualitatively. Additionally,we present **CarVal**, a datalog-based approach designed to infer multi-stage attack paths in IVNs and calculate risk values, thereby making TARA more efficient for automotive systems，By applying CarVal to five real vehicles, we performed extensive security analysis based on the generated attack paths and successfully exploited the corresponding attack chains in the newly gateway-segmented IVN, uncovering new automotive attack surfaces that previous research failed to cover, including xxx

- 对汽车网络安全专家进行访谈(访谈报告-[VehicleCyberSec/CarVal (github.com)](https://github.com/VehicleCyberSec/CarVal))
- 建立威胁数据库（An improved automotive threat database）
- CarVal-一种基于数据记录的方法，旨在推断 IVN(In vehicle network，车载网络) 中的多阶段攻击路径并计算风险值
- 根据路径进行安全分析，揭示了先前研究未能涵盖的新汽车攻击面

## 访谈过程

to-do	

## 访谈总结

确定了在汽车系统中开展安全活动的挑战

发现汽车网络安全具体法规的一系列限制（1.缺乏高质量的威胁数据库；2.TARA 缺乏高效工具）

## 改进威胁数据库

通过与15名专家访谈得到数据

## CarVal



## 入侵路径

- 绕过网关：从IVI（车载信息娱乐系统）浏览器到BCM（Body Control Module，车身控制模块）

- 从官方APP到汽车控制
- 通过车载以太网进行root权限获取
- 从互联网到汽车控制
- 从 IVI 恶意软件到汽车控制

## 总结（递进介绍）

1. 介绍访谈-揭示了当前具体挑战以及现有法规的局限性
2. 发现威胁
3. 构建了分层威胁数据库
4. 提出 CarVal，一种基于数据记录的方法，可以在 IVN 中生成多阶段攻击路径并计算风险值
5. 将CarVal应用到实际的车辆上成功利用新网关分段IVN中的相应攻击链

## 补充材料分析

```shell
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         2024/10/4     20:57                CarVal_Code
d-----         2024/10/4     20:57                Example
d-----         2024/10/4     20:57                Sup_Materials
-a----         2024/10/4     20:57           1641 README.md


    目录: C:\Users\WenTe\Desktop\CarVal\CarVal_Code


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2024/10/4     20:57           6726 carval_infer.sh #sh脚本
-a----         2024/10/4     20:57          10910 input_IVN.P
-a----         2024/10/4     20:57          13788 interaction_rules.P
-a----         2024/10/4     20:57           3534 risk_assessment.py #代码


    目录: C:\Users\WenTe\Desktop\CarVal\Example #论文提到的例子


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2024/10/4     20:57            741 AttackGraph.dot
-a----         2024/10/4     20:57           9955 AttackGraph.eps
-a----         2024/10/4     20:57          17225 risk_assess_output.pdf #输出的路径，下同
-a----         2024/10/4     20:57          79283 risk_assess_output.png


    目录: C:\Users\WenTe\Desktop\CarVal\Sup_Materials #补充材料


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2024/10/4     20:57          83923 Codebook.pdf #补充一些名词
-a----         2024/10/4     20:57         479032 Insights_and_Database.pdf 补充
-a----         2024/10/4     20:57          59826 Interview_protocol.pdf #采访的问题
-a----         2024/10/4     20:57            449 README.md 
-a----         2024/10/4     20:57        1201580 Security_Report.pdf #
```

## 对项目的借鉴点

汽车网络攻击路径对航电的网络攻击路径具有参考意义

建立模型与数据库

补充材料的编写


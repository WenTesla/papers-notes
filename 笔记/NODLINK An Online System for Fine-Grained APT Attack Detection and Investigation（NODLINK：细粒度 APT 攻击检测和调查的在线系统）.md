# NODLINK: An Online System for Fine-Grained APT Attack Detection and Investigation（NODLINK：细粒度 APT 攻击检测和调查的在线系统）

## 全文摘要

NODLINK-首个在不牺牲检测粒度的情况下保持高检测精度的在线检测系统 

提出新颖的内存缓存设计、高效的攻击筛选方法以及新的STP近似算法 



[开源版本](https://github.com/Nodlink/Simulated-Data)

## 基于来源的APT检测的工作流程

攻击候选者攻击

起源图构建

全面检测

## 专有名词

- **APT（Advanced Persistent Threat）高级持续攻击**
- **Endpoint Detection and Response (EDR) 端点检测与响应**

## 基于STP的APT攻击检测的挑战及应对

如何检测长期问题->STP 需要提前了解整个来源图->内存中保存保留所有来源数据遭遇内存瓶颈->提出一种新颖的内存缓存设计

如何在STP中高效地识别终端->设计一个需要最少计算的 IDF 加权三层变分自动编码器 (VAE)

当前的 STP 近似算法对于 APT 攻击检测仍然不够有效->开发一种面向重要性的贪婪算法用于在线 STP 优化，以有限的竞争比实现低计算复杂度

## 背景

### 出处分析

由于规则集不完整，基于规则的系统节点精度较低

### 在线斯坦纳树问题

作者提出算法

### 威胁模型



## 设计细节

### 内存缓存

< srcid, dstid, attr >源Id，目的Id，属性



### 终端识别

Nodlink识别下述三种终端

- 启动进程的命令行（命令行）
- 进程访问的文件（文件）
- 进程访问的IP地址（网络）

终端识别包括两个步骤：首先，它**根据节点级特征将流程节点嵌入到数值向量**中。其次，它使用机器学习模型来检测异常

### 跳跃集建设？

## 环境测试

- Open-World Experiment
- Close-World Experiment

## 介绍问题与回答问题

提供数据

提出攻击的路径（图画的真好👍）

## 总结

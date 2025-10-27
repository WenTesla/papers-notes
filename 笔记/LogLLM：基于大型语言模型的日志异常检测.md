# LogLLM：基于大型语言模型的日志异常检测

## 背景

- 传统深度学习方法：分为基于重构（如 LSTM、Autoencoder）和基于二分类（如 CNN、Bi-LSTM）两类，均难以捕捉日志语义。
- LLM 相关方法：提示工程类方法（如利用 ChatGPT 零样本检测）难以适配特定数据集，微调类方法存在语义理解不足、内存溢出、LLM 利用不充分等问题。

为了应对上述挑战，我们提出了LogLLM，这是一种利用大语言模型（LLMs）的新型基于日志的异常检测框架。与依赖日志解析器进行模板提取的传统方法不同，LogLLM使用正则表达式对日志消息进行预处理，从而简化了整个流程。LogLLM是一种基于微调的方法。

### 提示工程的方法（Prompt engineering-based methods [7], [29]–[31]）

**完全依赖 LLMs 的内部知识**实现异常检测，无需对模型进行针对特定日志数据集的微调，仅通过设计合理的提示模板（Prompt Templates）引导 LLMs 输出判断结果。从模型选型来看，它们通常采用**Transformer 解码器类 LLMs**（如 ChatGPT），这类模型在自然语言理解、零样本 / 少样本任务中具备较强能力，能够基于提示中的指令和日志信息进行推理，判断日志序列是否异常。

**数据集定制化能力弱**的问题：由于这类方法依赖 LLMs 的通用内部知识，未针对特定日志数据集的特征（如日志格式、领域术语、异常模式）进行适配，导致在部分特定数据集上难以达到理想的检测性能。

## 贡献

我们提出了LogLLM，这是一种利用大语言模型（LLMs）的新型基于日志的异常检测框架。本研究首次尝试同时采用基于Transformer编码器和基于Transformer解码器的大语言模型（具体为BERT和Llama）来进行基于日志的异常检测。

我们提出了一种新颖的三阶段程序，以优化深度模型内不同组件的训练和协调，从而同时提升性能和适应性。

我们在四个公开可用的真实世界数据集上进行了大量实验，结果表明LogLLM取得了优异的性能。

## 框架

![img](https://cdn.xljsci.com/literature/176848499/page4/i9cd31.png)





## 效果图

![img](https://cdn.xljsci.com/literature/176848499/page7/uzzf05.png)

![img](https://cdn.xljsci.com/literature/176848499/page7/w0u9nr.png)

![img](https://cdn.xljsci.com/literature/176848499/page8/cu5oya.png)

![img](https://cdn.xljsci.com/literature/176848499/page8/ah1wom.png)

![img](https://cdn.xljsci.com/literature/176848499/page9/b04u3z.png)
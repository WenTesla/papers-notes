# NeuralLog：Log-based Anomaly Detection Without Log Parsing





## 贡献

1. 我们对日志解析错误进行了实证研究。我们发现现有的基于日志的异常检测方法受到OOV词和语义误解引入的日志解析错误的不利影响
2. 我们提出了NeuralLog，这是一种新颖的基于深度学习的方法，无需日志解析就能检测系统异常。NeuralLog利用了BERT（一种广泛使用的预训练语言表示模型）来对语义进行编码。



> OOV 指Out-of-Vocabulary（未登录词），即那些未出现在模型训练阶段构建的词汇表中的词语。



## 日志解析的影响

### OVV

为了确定OOV词，我们首先**按日志的时间戳对日志消息进行排序**，并利用前面的P%（根据日志的时间戳）作为训练数据，其余的作为测试数据。

然后，我们通过空格字符将每个训练日志消息拆分为一组标记，并从这些标记中构建词汇表。OOV词是测试数据中词汇表中不存在的那些词。在本节中，我们将训练数据的百分比从20%增加到80%。然后我们计算OOV词在所有拆分中的比例。

<img src="https://cdn.xljsci.com/literature/173891521/page3/57bcq4.png" alt="img" style="zoom:67%;" />

<img src="https://cdn.xljsci.com/literature/173891521/page3/3kyx4d.png" alt="img" style="zoom:67%;" />

原因：

1. 许多日志事件仅在特定时期出现。
2. 日志事件的分布是不平衡的。
3. OOV词会导致日志解析错误并导致许多额外的日志事件

### 解析错误

<img src="https://cdn.xljsci.com/literature/173891521/page4/es5p94.png" alt="img" style="zoom:67%;" />

对于案例1，日志消息中的参数被错误识别为关键字，并被包含在日志解析器生成的日志模板中，从而导致**许多额外的日志事件。**我们将每条日志消息的解析模板与BGL数据集的真实情况进行了比较。如果一个模板包含的关键字多于真实情况，则认为其解析错误，属于额外的日志事件。图5中案例1的**两条日志消息本应对应一个日志模板**，却被解析成了两个不同的日志模板。

对于情况2，日志解析后可能会移除日志消息中的一些关键关键词，导致**不同的日志消息被解析为一个日志事件**。图5展示了这种情况的一个示例。两条不同的日志消息被解析为相同的日志事件“machine check ∗”。然而，其中一条表示正常行为（即“machine check enable”），另一条则表示系统异常（即“machine check interrupt”）。这类错误使得检测模型难以仅根据日志事件来区分日志是正常的还是异常的。

![图7](https://cdn.xljsci.com/literature/173891521/page4/l5yips.png)

上图展示了四个日志解析器所引入的情况2解析错误的示例。每个示例都显示了一条正常日志和一条异常日志被解析为相同的日志事件。日志事件中缺失了诸如登录失败原因等有价值的信息（即图7(a)），这导致了许多错误的检测结果。

<img src="https://cdn.xljsci.com/literature/173891521/page4/7zhpqk.png" alt="img" style="zoom:67%;" />

上图显示了四种日志解析器在两个数据集上产生的额外日志事件的百分比。例如，使用Drain时，BGL数据集上约有80%的额外日志事件，Thunderbird数据集上约有72%的额外日志事件。





## 架构

<img src="https://cdn.xljsci.com/literature/173891521/page6/2xpscc.png" alt="img" style="zoom:67%;" />

### 预处理

将一条日志消息分词为一组单词标记。

我们使用日志系统中的常见分隔符（即空格、冒号、逗号等）来拆分日志消息。然后，将每个大写字母转换为小写字母，并从单词集中移除所有非字符标记。**这些非字符包括运算符、标点符号和数字**。之所以移除这类标记，是因为它们在日志消息中通常代表变量，不具有信息价值。

例如，原始日志消息

```
081109 205931 13 INFO dfs.DataBlockScanner: Verification succeeded for blk -4980916519894289629
```

首先会根据常见分隔符拆分为一组单词，然后去除非字符这些标记被排除在集合之外。最后，得到了一组单词{info、dfs、datablockscanner、verification、succeeded}。

### 神经表示

#### 子词分词

使用**wordPiece**进行分词。

> WordPiece 是一种**子词级（subword-level）分词算法**，核心目标是通过将词汇拆分为更小的、高频出现的子词单元，解决传统分词（如基于空格、词根的分词）在处理罕见词（OOV 词）时的局限性 —— 既减少词汇表规模，又能保留未见过词汇的语义信息，确保模型能对 OOV 词进行有效编码

#### 消息表示

> NeuralLog将每条日志消息转换为**一组单词和子词**。传统上，日志内容的单词会通过Word2Vec[42]进一步转换为向量，然后基于这些词向量计算每个句子的表示向量。然而，Word2Vec会为同一个词生成相同的嵌入。在很多情况下，一个词的含义会因其位置和上下文而有所不同。
>
> BERT[38]是一种最新的深度学习表示模型，已在庞大的自然语言语料库上进行了预训练。在我们的研究中，**我们利用预训练BERT的特征提取功能来获取日志消息的语义。**

在进行分词后，单词和子词的集合会被传入BERT模型，并被编码为固定维度的向量表示。NeuralLog采用了BERT基础模型[43]，该模型包含12层Transformer编码器，每层Transformer有768个隐藏单元。

每层都会为日志消息中的每个子词生成嵌入。使用BERT最后一个编码器层生成的词嵌入。然后，日志消息的嵌入被计算为其对应词嵌入的平均值。由于任何未出现在词汇表中的词（即OOV词）都会被拆分为子词，BERT能够基于子词集合的含义学习这些OOV词的表示向量。此外，位置嵌入层使BERT能够根据词在日志消息中的上下文捕捉其表示。BERT还包含自注意力机制，能够有效衡量句子中每个词的重要性。

#### Transformer





## 效果

![img](https://cdn.xljsci.com/literature/173891521/page7/amp0wd.png)

#### HDFS

对于HDFS数据集，我们通过关联具有相同块ID的日志消息来构建日志序列，因为数据是按块标记的

#### Thunderbird，BGL，Spirit

按时间排序

采用20的时间窗口进行排序



### 消融实验

![img](https://cdn.xljsci.com/literature/173891521/page9/r3r6ep.png)

NeuralLog-Index：日志模板的索引由Drain [24]获取，被简单编码为**数值向量**后传入Transformer模型进行异常检测。NeuralLog的其余部分保持不变。

NeuralLog-Template：我们利用BERT将Drain [24]生成的**日志模板编码为语义向量**。然后，我们将这些语义向量输入Transformer模型进行异常检测。NeuralLog的其余部分保持不变。

![img](https://cdn.xljsci.com/literature/173891521/page10/gbsyfk.png)

NeuralLog-Word2Vec：我们使用预训练的Word2vec模型[31]来生成日志消息的嵌入。那些不在词汇表中的单词会从日志消息中移除。然后，嵌入向量被传递到Transformer模型以检测异常。

NeuralLog-NoWordPiece：我们从模型中排除了WordPiece分词器（见图10）。经过预处理的日志消息直接输入BERT模型以获取语义向量。然后，这些向量被输入Transformer模型进行异常检测。通过这种方式，词汇表中不存在的未登录词（OOV）会被移除，而不是被拆分为子词。


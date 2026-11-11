# 使用机器学习进行翻译和情感分析

在之前的课程中，你学习了如何使用 `TextBlob` 构建一个基础的机器人。`TextBlob` 是一个库，它在幕后嵌入了机器学习技术，用于执行基本的自然语言处理任务，例如名词短语提取。计算语言学中的另一个重要挑战是准确地将一个语言的句子翻译成另一种语言。

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

翻译是一个非常困难的问题，因为世界上有成千上万种语言，每种语言都有非常不同的语法规则。一种方法是将一种语言（例如英语）的正式语法规则转换为一种与语言无关的结构，然后通过转换回另一种语言来完成翻译。这种方法的步骤如下：

1. **识别**。识别或标记输入语言中的单词，例如名词、动词等。
2. **创建翻译**。以目标语言的格式直接翻译每个单词。

### 示例句子：英语到爱尔兰语

在“英语”中，句子 _I feel happy_ 是三个单词，顺序为：

- **主语** (I)
- **动词** (feel)
- **形容词** (happy)

然而，在“爱尔兰语”中，同样的句子有非常不同的语法结构——像“happy”或“sad”这样的情感被表达为“在你身上”。

英语短语 `I feel happy` 在爱尔兰语中是 `Tá athas orm`。一个*字面*翻译是 `Happy is upon me`。

一个讲爱尔兰语的人翻译成英语时会说 `I feel happy`，而不是 `Happy is upon me`，因为他们理解句子的含义，即使单词和句子结构不同。

在爱尔兰语中，这句话的正式顺序是：

- **动词** (Tá 或 is)
- **形容词** (athas 或 happy)
- **主语** (orm 或 upon me)

## 翻译

一个简单的翻译程序可能只翻译单词，而忽略句子结构。

✅ 如果你作为成年人学习了第二（或第三甚至更多）语言，你可能一开始会在脑海中用母语思考，将一个概念逐字翻译成第二语言，然后说出你的翻译。这类似于简单翻译计算机程序的工作方式。要达到流利程度，重要的是要超越这个阶段！

简单翻译会导致糟糕（有时甚至是搞笑）的误译：`I feel happy` 字面翻译成爱尔兰语是 `Mise bhraitheann athas`。这意味着（字面上）`me feel happy`，但这不是一个有效的爱尔兰语句子。尽管英语和爱尔兰语是两个邻近岛屿上使用的语言，但它们是非常不同的语言，语法结构也不同。

> 你可以观看一些关于爱尔兰语言传统的视频，例如 [这个](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### 机器学习方法

到目前为止，你已经了解了自然语言处理的正式规则方法。另一种方法是忽略单词的含义，而是*使用机器学习来检测模式*。如果你有大量的文本（*语料库*）或原始语言和目标语言的文本（*语料*），这种方法在翻译中可能会奏效。

例如，考虑《傲慢与偏见》的情况，这是一本由简·奥斯汀于1813年写的著名英语小说。如果你查阅这本书的英语版本和人类翻译的*法语*版本，你可以检测到一种语言中的短语在另一种语言中被*习惯性地*翻译。这就是你接下来要做的。

例如，当英语短语 `I have no money` 被字面翻译成法语时，它可能变成 `Je n'ai pas de monnaie`。“Monnaie” 是一个棘手的法语“假同源词”，因为“money”和“monnaie”并不是同义词。一个人类可能会做出更好的翻译，即 `Je n'ai pas d'argent`，因为它更好地传达了你没有钱的意思（而不是“零钱”，这是“monnaie”的意思）。

![monnaie](images/monnaie.png)

> 图片由 [Jen Looper](https://twitter.com/jenlooper) 提供

如果一个机器学习模型有足够的人工翻译来构建模型，它可以通过识别之前由精通两种语言的专家翻译的文本中的常见模式来提高翻译的准确性。

### 练习 - 翻译

你可以使用 `TextBlob` 来翻译句子。试试《傲慢与偏见》的著名第一句：

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` 的翻译效果相当不错：“C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!”。

可以说，`TextBlob` 的翻译实际上比1932年由 V. Leconte 和 Ch. Pressoir 翻译的法语版本更精确：

“C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles。”

在这种情况下，由机器学习支持的翻译比人类翻译更好，因为后者为了“清晰”而不必要地在原作者的文字中添加了额外的内容。

> 这是怎么回事？为什么 `TextBlob` 的翻译如此出色？实际上，它在幕后使用了 Google Translate，这是一种复杂的人工智能，能够解析数百万个短语以预测最适合当前任务的字符串。这完全是自动化的，你需要互联网连接才能使用 `blob.translate`。

✅ 尝试更多句子。机器学习翻译和人工翻译哪个更好？在哪些情况下？

## 情感分析

机器学习在情感分析领域也表现得非常出色。一种非机器学习的方法是识别“积极”和“消极”的单词和短语。然后，给定一段新的文本，计算积极、消极和中性单词的总值，以确定整体情感。

这种方法很容易被欺骗，就像你在 Marvin 任务中看到的那样——句子 `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` 是一个讽刺性的消极情感句子，但简单的算法会检测到“great”、“wonderful”、“glad”是积极的，“waste”、“lost”和“dark”是消极的。整体情感被这些矛盾的单词所影响。

✅ 停下来想一想，作为人类说话者，我们如何表达讽刺。语调的变化起到了很大的作用。试着用不同的方式说“Well, that film was awesome”，看看你的声音如何传达意义。

### 机器学习方法

机器学习方法是手动收集消极和积极的文本——例如推文、电影评论，或者任何带有评分*和*书面意见的内容。然后可以将 NLP 技术应用于意见和评分，从而发现模式（例如，积极的电影评论中“奥斯卡级”这个短语出现的频率比消极电影评论中高，或者积极的餐厅评论中“美食”出现的频率比“恶心”高）。

> ⚖️ **示例**：如果你在一个政治家的办公室工作，并且有一项新的法律正在讨论，选民可能会写邮件支持或反对这项新法律。假设你的任务是阅读这些邮件并将它们分为两类：*支持*和*反对*。如果邮件很多，你可能会因为试图阅读所有邮件而感到不堪重负。如果有一个机器人可以阅读所有邮件，理解它们并告诉你每封邮件属于哪个类别，那不是很好吗？
> 
> 一种实现方法是使用机器学习。你可以用一部分*反对*邮件和一部分*支持*邮件来训练模型。模型会倾向于将某些短语和单词与反对方或支持方关联起来，*但它不会理解任何内容*，只会知道某些单词和模式更可能出现在反对或支持邮件中。你可以用一些未用于训练模型的邮件进行测试，看看它是否得出了与你相同的结论。然后，一旦你对模型的准确性感到满意，你就可以处理未来的邮件，而无需逐一阅读。

✅ 这个过程是否类似于你在之前课程中使用的过程？

## 练习 - 情感句子

情感通过*极性*从 -1 到 1 来衡量，-1 表示最消极的情感，1 表示最积极的情感。情感还通过 0 到 1 的分数来衡量客观性（0）和主观性（1）。

再看一眼简·奥斯汀的《傲慢与偏见》。文本可以在 [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) 找到。以下示例展示了一个简短的程序，它分析了书中第一句和最后一句的情感，并显示其情感极性和主观性/客观性分数。

你应该使用 `TextBlob` 库（如上所述）来确定 `sentiment`（你不需要自己编写情感计算器）来完成以下任务。

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

你会看到以下输出：

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## 挑战 - 检查情感极性

你的任务是使用情感极性来确定《傲慢与偏见》中绝对积极的句子是否多于绝对消极的句子。对于此任务，你可以假设极性分数为 1 或 -1 的句子是绝对积极或消极的。

**步骤：**

1. 从 Project Gutenberg 下载一份《傲慢与偏见》的 [副本](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) 作为 .txt 文件。删除文件开头和结尾的元数据，仅保留原始文本。
2. 在 Python 中打开文件并将内容提取为字符串。
3. 使用书的字符串创建一个 TextBlob。
4. 在循环中分析书中的每个句子：
   1. 如果极性为 1 或 -1，将句子存储在一个数组或列表中，分别存储积极或消极的消息。
5. 最后，分别打印出所有积极句子和消极句子，以及它们的数量。

这里是一个 [示例解决方案](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb)。

✅ 知识检查

1. 情感是基于句子中使用的单词，但代码是否*理解*这些单词？
2. 你认为情感极性准确吗？换句话说，你是否*同意*这些分数？
   1. 特别是，你是否同意或不同意以下句子的绝对**积极**极性：
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. 以下三个句子被评分为绝对积极情感，但仔细阅读后，它们并不是积极句子。为什么情感分析认为它们是积极句子？
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. 你是否同意或不同意以下句子的绝对**消极**极性：
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ 任何简·奥斯汀的爱好者都会理解，她经常在书中批评英国摄政时期社会中更荒谬的方面。《傲慢与偏见》的主角伊丽莎白·班内特是一个敏锐的社会观察者（就像作者一样），她的语言通常充满了深意。甚至故事中的爱情对象达西先生也注意到伊丽莎白的俏皮和戏谑的语言使用：“我有幸认识你足够久，知道你偶尔会发表一些实际上并非你真实观点的意见，并从中获得极大的乐趣。”

---

## 🚀挑战

你能通过从用户输入中提取其他特征来让 Marvin 更加出色吗？

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学
从文本中提取情感有很多方法。想想可能会利用这种技术的商业应用。再想想它可能出错的情况。阅读更多关于分析情感的复杂企业级系统，例如 [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott)。测试上面的一些《傲慢与偏见》的句子，看看它是否能检测出细微差别。

## 作业

[诗意许可](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保翻译的准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。我们不对因使用此翻译而产生的任何误解或误读承担责任。
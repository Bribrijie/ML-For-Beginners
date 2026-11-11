# 机器学习中的聚类模型

聚类是一种机器学习任务，旨在寻找彼此相似的对象并将它们分组到称为“聚类”的组中。与机器学习中的其他方法不同，聚类是自动进行的，实际上可以说它是监督学习的反面。

## 地区主题：针对尼日利亚观众音乐品味的聚类模型 🎧

尼日利亚的观众拥有多样化的音乐品味。通过从 Spotify 抓取的数据（灵感来源于[这篇文章](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)），让我们来看看尼日利亚流行的一些音乐。这份数据集包括关于各种歌曲的“舞蹈性”评分、“声学性”、响度、“语音性”、流行度和能量的相关数据。发现这些数据中的模式将会非常有趣！

![唱盘](../translated_images/zh-CN/turntable.f2b86b13c53302dc.webp)

> 图片由 <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> 提供，来自 <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
在这一系列课程中，你将学习使用聚类技术分析数据的新方法。聚类特别适用于数据集缺乏标签的情况。如果数据集有标签，那么你在之前课程中学到的分类技术可能会更有用。但在需要对无标签数据进行分组的情况下，聚类是发现模式的绝佳方法。

> 有一些实用的低代码工具可以帮助你学习如何使用聚类模型。试试 [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) 来完成这个任务。

## 课程

1. [聚类简介](1-Visualize/README.md)
2. [K-Means 聚类](2-K-Means/README.md)

## 致谢

这些课程由 [Jen Looper](https://www.twitter.com/jenlooper) 倾情创作，并由 [Rishit Dagli](https://rishit_dagli) 和 [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) 提供了有益的审阅。

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) 数据集来源于 Kaggle，由 Spotify 抓取。

在创建本课程时，以下 K-Means 示例提供了帮助，包括这个 [鸢尾花探索](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)、这个[入门笔记本](https://www.kaggle.com/prashant111/k-means-clustering-with-python)，以及这个[假设的 NGO 示例](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)。

---

**免责声明**：  
本文档使用AI翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 进行翻译。尽管我们努力确保翻译的准确性，但请注意，自动翻译可能包含错误或不准确之处。原始语言的文档应被视为权威来源。对于关键信息，建议使用专业人工翻译。我们不对因使用此翻译而产生的任何误解或误读承担责任。
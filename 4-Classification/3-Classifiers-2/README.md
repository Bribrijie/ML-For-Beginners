# Cuisine classifiers 2

在第二节分类课程中，您将探索更多对数值数据进行分类的方法。您还将了解选择不同分类器的影响。

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

### 先决条件

假设您已经完成了之前的课程，并且在该 4 节课程文件夹根目录的 `data` 文件夹中有一个清洗好的数据集文件 _cleaned_cuisines.csv_。

### 准备工作

我们已经加载了包含清洗后数据集的 _notebook.ipynb_ 文件，并将其分拆成 X 和 y 两个数据框，准备好进行模型构建。

## 分类图谱

之前，您通过微软的分类小抄了解了分类数据时的各种选择。Scikit-learn 提供了一个类似但更细致的分类小抄，可以进一步帮助收窄你的估计器（分类器）选择：

![ML Map from Scikit-learn](../../translated_images/zh-CN/map.e963a6a51349425a.webp)
> 提示: [在线查看该图谱](https://scikit-learn.org/stable/tutorial/machine_learning_map/) 并点击路径查看文档。

### 计划

一旦您对数据有了清晰的理解，这张图谱非常有帮助，您可以沿着路径做决策：

- 我们有超过 50 个样本
- 我们想预测一个类别
- 我们有标签数据
- 我们有少于 10 万个样本
- ✨ 我们可以选择线性 SVC
- 如果不行，因为我们有数值数据
    - 可以尝试 ✨ KNeighbors 分类器
      - 如果不行，再尝试 ✨ SVC 和 ✨ 集成分类器

这是一个很有帮助的思路路径。

## 练习 - 拆分数据

沿着这个路径，先导入一些需要使用的库。

1. 导入所需的库：

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. 拆分训练集和测试集：

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
    ```

## 线性 SVC 分类器

支持向量机（SVC）是支持向量机家族的机器学习技术之一（后文将深入了解）。此方法中，您可以选用“核函数”来决定如何聚类标签。“C”参数指正则化，控制参数对模型的影响。核函数可以是 [多种](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)之一；这里我们将其设为 'linear' 以使用线性 SVC。概率默认是 'false'；这里设置为 'true' 以收集概率估计。为了打乱数据并得到概率，我们将随机状态设为 '0'。

### 练习 - 应用线性 SVC

先创建一个分类器数组，后续测试将逐步添加。

1. 先创建一个线性 SVC:

    ```python
    C = 10
    # 创建不同的分类器。
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. 用线性 SVC 训练模型并打印报告：

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    结果相当不错：

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## K-邻近分类器

K-邻近属于邻居类机器学习方法，可用于有监督和无监督学习。此方法定义预设点数，数据聚集于这些点周围，从而预测数据的一般化标签。

### 练习 - 应用 K-邻近分类器

先前的分类器效果不错，并且适配数据，但我们可能能得到更好的准确率。试试 K-邻近分类器。

1. 在分类器数组中添加一行（在线性 SVC 项后添加逗号）：

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    结果稍差一点：

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ 学习关于 [K-邻近](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## 支持向量分类器

支持向量分类器是 [支持向量机](https://wikipedia.org/wiki/Support-vector_machine) 家族的机器学习方法，用于分类与回归任务。支持向量机“将训练样本映射为空间中的点”，以最大化两类别之间的距离。之后的数据映射进空间以预测类别。

### 练习 - 应用支持向量分类器

尝试用支持向量分类器提升准确率。

1. 在 K-邻近项后加逗号，然后添加此行：

    ```python
    'SVC': SVC(),
    ```

    结果很不错！

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ 学习关于 [支持向量](https://scikit-learn.org/stable/modules/svm.html#svm)

## 集成分类器

我们沿路径尝试到最终，尽管之前测试效果不错，还是试试“集成分类器”，具体是随机森林和 AdaBoost：

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

结果非常好，特别是随机森林：

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ 学习关于 [集成分类器](https://scikit-learn.org/stable/modules/ensemble.html)

这种机器学习方法“结合多个基估计器的预测”来提升模型质量。在我们的示例中，使用了随机树和 AdaBoost。

- [随机森林](https://scikit-learn.org/stable/modules/ensemble.html#forest)，一种平均方法，构建大量注入随机性的“决策树”的森林以避免过拟合。n_estimators 参数设为树的数量。

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) 拟合一个分类器到数据集，然后拟合该分类器的副本到同一数据集。它关注分类错误样本的权重，并调整后续分类器的拟合以纠正错误。

---

## 🚀挑战

每种技术都有大量参数可调。研究它们各自的默认参数，思考调整这些参数对模型质量意味着什么。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

这些课程中有大量术语，花点时间复习 [这份术语表](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) 吧！

## 作业

[参数练习](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。虽然我们努力确保准确性，但请注意自动翻译可能包含错误或不准确之处。原始语言的文档应被视为权威来源。对于关键信息，建议使用专业人工翻译。我们不对因使用本翻译而产生的任何误解或误释承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->
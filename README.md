# breast-cancer-wisconsin
通过利用sklearn，对良、恶性乳腺癌肿瘤做逻辑回归预测 2.数据集由UCI提供，共有11项数据指标，具体数据：[https://archive.ics.uci.edu/ml/](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data)
Sklearn，全称为 Sklearn，是一个基于 Python 的开源机器学习库。

Sklearn 是一个功能强大、易于使用的机器学习库，提供了从数据预处理到模型评估的一整套工具。

Sklearn 构建在 NumPy 和 SciPy 基础之上，因此它能够高效地处理数值计算和数组操作。

Sklearn 适用于各类机器学习任务，如分类、回归、聚类、降维等。

Sklearn 凭借其简洁而一致的 API，Sklearn 已经成为了机器学习爱好者和专家的必备工具之一。
在 Sklearn 中，机器学习的流程遵循一定的模式：数据加载、数据预处理、训练模型、评估模型 和 调优模型。
具体工作流程如下：

数据加载：使用 Sklearn 或其他库加载数据集，例如通过 datasets.load_iris() 加载经典的鸢尾花数据集，或使用 train_test_split() 分割数据。
数据预处理：根据数据的类型，可能需要进行标准化、去噪、缺失值填充等操作。
选择算法和训练模型：选择适合的算法（如逻辑回归、支持向量机等），使用 .fit() 方法对模型进行训练。
模型评估：使用交叉验证或单一训练/测试集来评估模型的准确性、召回率、F1分数等性能指标。
模型优化：使用网格搜索（GridSearchCV）或随机搜索（RandomizedSearchCV）对模型进行超参数优化，提高模型性能。
Sklearn 支持的机器学习任务
Sklearn 提供了丰富的工具，支持以下几类机器学习任务：

监督学习（Supervised Learning）：
分类问题：预测数据的类别（例如，电子邮件垃圾邮件分类、图像分类、疾病预测等）。
回归问题：预测连续值（例如，房价预测、股票价格预测等）。
无监督学习（Unsupervised Learning）：
聚类问题：将数据分组为不同的类群（例如，客户分群、文档聚类等）。
降维问题：将高维数据投影到低维空间，便于可视化或减少计算复杂度（例如，PCA、t-SNE）。
半监督学习（Semi-supervised Learning）：部分数据是带标签的，部分数据是无标签的，模型尝试从这些数据中提取信息。
强化学习（Reinforcement Learning）：虽然 Sklearn 主要专注于监督和无监督学习，但也有一些相关的工具，可以用来处理强化学习问题。
1.roc_auc_score()（ROC AUC 分数）
作用：衡量模型的排序能力（即模型对正负样本的区分能力）。

计算方式：

基于预测概率（如 lr.predict_proba(X)[:, 1]），计算 ROC 曲线下的面积（AUC）。

取值范围：0.5（随机猜测）到 1.0（完美分类）。

适用场景：

适用于二分类问题（多分类需要 multi_class 参数调整）。

适用于类别不平衡的数据集（因为 ROC AUC 不受类别分布影响）。

关注的是模型的整体排序能力（正样本比负样本得分高的概率）。

示例：

python

y_proba = lr.predict_proba(X_test)[:, 1]  # 取正类的概率
roc_auc = roc_auc_score(y_test, y_proba)
print(roc_auc)  # 输出 0.0~1.0 的值
2. lr.score()（模型默认评分）
作用：计算模型的准确率（Accuracy）（即正确预测的比例）。

计算方式：

默认使用 accuracy_score(y_true, y_pred)，即：

Accuracy
=
正确预测数
总样本数
Accuracy= 
总样本数
正确预测数
​
 
如果是 LogisticRegression，lr.score(X, y) 等价于：

python

y_pred = lr.predict(X)
accuracy = (y_pred == y).mean()
适用场景：

适用于类别平衡的数据集（如果类别不平衡，准确率可能误导）。

关注的是最终分类结果，而非概率。

示例：

python
复制
accuracy = lr.score(X_test, y_test)
print(accuracy)  # 输出 0.0~1.0 的值

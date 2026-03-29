# DSCI 6800 A1

这个项目是 DSCI 6800 Assignment 1 的代码框架。

## 运行环境

- Python 3
- `numpy`

如果本机还没有安装 `numpy`，可以先运行：

```bash
pip install numpy
```

## 如何运行

在项目根目录下打开终端，运行：

```bash
python train_and_test.py <classifier_type> <dataset_name>
```

## 当前支持的分类器

- `naive_bayes`
- `logistic_regression`
- `decision_tree`

## 当前支持的数据集

- `iris`
- `congress`
- `monks1`
- `monks2`
- `monks3`

## 运行示例

```bash
python train_and_test.py naive_bayes iris
python train_and_test.py logistic_regression congress
python train_and_test.py decision_tree monks1
```

## 当前状态说明

- 目前主程序入口已经搭好
- 训练集和测试集都会输出 accuracy、precision、recall
- 现在只有整体框架，具体分类器实现还需要继续补

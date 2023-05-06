# 导入必备工具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# 设置显示风格
plt.style.use('fivethirtyeight') 

# 分别读取训练tsv和验证tsv
# 分隔符采用\t
train_data = pd.read_csv("./cn_data/train.tsv", sep="\t")
valid_data = pd.read_csv("./cn_data/dev.tsv", sep="\t")


# 获得训练数据标签数量分布
# 不要忘记写x
sns.countplot(x='label', data=train_data)
plt.title("train_data")
plt.show()


# 获取验证数据标签数量分布
sns.countplot(x='label', data=valid_data)
plt.title("valid_data")
plt.show()

# 在训练数据中添加新的句子长度列, 每个元素的值都是对应的句子列的长度
# sentence是标签(还有一个是label)
print(train_data["sentence"])
train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))
print(type(train_data["sentence_length"]))
print(train_data["sentence_length"])

# 绘制句子长度列的数量分布图
sns.countplot(x="sentence_length", data=train_data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.show()

# 绘制dist长度分布图
sns.distplot(train_data["sentence_length"])

# 主要关注dist长度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()


# 在验证数据中添加新的句子长度列, 每个元素的值都是对应的句子列的长度
valid_data["sentence_length"] = list(map(lambda x: len(x), valid_data["sentence"]))

# 绘制句子长度列的数量分布图
sns.countplot(x="sentence_length", data=valid_data)

# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.show()

# 绘制dist长度分布图
sns.distplot(valid_data["sentence_length"])

# 主要关注dist长度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.show()

# 绘制训练集长度分布的散点图
sns.stripplot(y='sentence_length',x='label',data=train_data)
plt.show()

# 绘制验证集长度分布的散点图
sns.stripplot(y='sentence_length',x='label',data=valid_data)
plt.show()


# 导入jieba用于分词
# 导入chain方法用于扁平化列表
import jieba
from itertools import chain

# 进行训练集的句子进行分词, 并统计出不同词汇的总数
train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data["sentence"])))
print("训练集共包含不同词汇总数为：", len(train_vocab))

# 进行验证集的句子进行分词, 并统计出不同词汇的总数
valid_vocab = set(chain(*map(lambda x: jieba.lcut(x), valid_data["sentence"])))
print("训练集共包含不同词汇总数为：", len(valid_vocab))
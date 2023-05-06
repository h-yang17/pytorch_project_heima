import jieba
import hanlp
from hanlp.utils.lang.en.english_tokenizer import tokenize_english
import jieba.posseg as pseg

content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
cut = jieba.cut(content, cut_all=False)
print(cut)
lcut = jieba.lcut(content, cut_all=False)
print(lcut)
# 全模式把所有能分词的都分割出来
jieba_lcut = jieba.lcut(content, cut_all=True)
print(jieba_lcut)

content = "煩惱即是菩提，我暫且不提"
l = jieba.lcut(content, cut_all=False)
print(l)

lcut1 = jieba.lcut("八一双鹿更名为八一南昌篮球队！")
print(lcut1)

jieba.load_userdict("./user_dict.txt")
lcut1 = jieba.lcut("八一双鹿更名为八一南昌篮球队！")
print(lcut1)
#
# tokenizer = hanlp.load('CTB6_CONVSEG')
# tokenizer("工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作")

tokenizer = tokenize_english
list_list = tokenizer('Mr. Hankcs bought hankcs.com for 1.5 thousand dollars.')
print(list_list)

# recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
# recognizer1 = recognizer(list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。'))
# print(recognizer1)


# recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN))
# recognizer1 = recognizer(["President", "Obama", "is", "speaking", "at", "the", "White", "House"])
# print(recognizer1)


pseg_lcut = pseg.lcut("我爱北京天安门")
print(pseg_lcut)

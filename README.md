## 基于bert的多标签文本分类

建立的目的是做个记录，虽然bert刚出来的时候就关注了，好像吓到了很多人，但一直没搞过，现在轮到我震惊了。

实际使用可以使用开源项目transformers，封装的更好

参考于：

<https://www.kaggle.com/javaidnabi/toxic-comment-classification-using-bert/> 

# 需要安装

bert-tensorflow

或者将谷歌bert代码放到项目内新建bert文件夹里

<https://github.com/google-research/bert> 

# 数据集下载

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

新建data文件夹，并将train.csv解压到里面即可

基本上都是在谷歌的bert上加个下游任务进行应用

# 模型下载

https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

新建bert_ckpt并解压到里面

## 使用

先写入成tfrecord

python data_load.py

然后训练并在训练过程中评估（我的电脑跑不动的）

python main.py --do_train=true --do_eval=true

最后可以预测，里面有个简单的预测案例

python main.py --do_predict=true


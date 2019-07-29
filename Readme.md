# CCKS 2019

# 系统依赖包

1、PyTorch 0.4.1 to 1.1.0
2、bert预训练模型---pip install pytorch-pretrained-bert
3、pip install scipy
4、pip install sklearn




# mysql数据库安装及知识库导入

mysql数据库安装参考地址：https://note.youdao.com/ynoteshare1/index.html?id=be2987b19c57709b1ac1da1f8314f219&type=note

知识库及相关数据的地址：https://pan.baidu.com/s/1MOv9PCTcALVIiodUP4bQ2Q 密码：hcu8

知识库导入方式：pkubase.ipynb


## Bert NER结果和暴力搜索的结果融合
所需文件：

暴力搜索得到的mention文件：系统3_规则/output_data/test_keywords_复现.txt

test集经Bert做NER的输出文件：系统1_模型训练/NER模型/ner/test_model_decode

运行jupyter：NER融合/实体识别和实体链接/实体识别的结合与实体链接.ipynb

得到最后的实体识别结果及实体链接结果：test_实体识别结果_加长串_baike_0728.pkl、test_实体链接_加长串_baike_top8_0728.pkl

# 系统一：模型系统
## 单个求解
### 二合一问题求解
进入文件夹：系统1_模型and规则/路径_答案查找/Step1_二合一求交集/
1. 运行Step1_二合一求交集.ipynb,提取出test问题集中的两跳问句及其相关路径：test二合一path.pkl

同时，生成性别限制的二合一问题及相关路径：paths_gender.pkl

2. 运行：Step2_生成输入得分计算模型的文件.ipynb，生成输入bert的文件

3. 计算最优路径

 --fn_in "bert_input_test_gender_path.pkl" --fn_out 'mention_test_gender_path.pkl' ,运行
```
python Step3_score_bert.py 
```
修改Step3_score_bert.py里主函数中的f和f1

 --fn_in "bert_input_test_二合一_path.pkl" --fn_out 'mention_test_二合一_path.pkl' ,运行
```
python Step3_score_bert.py 
```

4. 运行：Step4_二合一找答案.ipynb，生成答案文件：test_二合一_ans.pkl、test_gender_ans.pkl

### 三合一问题求解
进入文件夹：系统1_模型and规则/路径_答案查找/Step1_三合一求交集/

1. 运行Step2_三合一求交集.ipynb,提取出test问题集中的两跳问句及其相关路径：test三合一path.pkl

2. 运行：Step2_生成输入得分计算模型的文件.ipynb，生成输入bert的文件

3. 计算最优路径
```
python Step3_score_bert.py 
```

4. 运行：Step4_二合一找答案.ipynb，生成答案文件：test_三合一_ans.pkl


### 单跳问句求解
进入文件夹：系统1_模型and规则/路径_答案查找/Step3_单跳问句求路径答案/

1. 对除去二合一、三合一的句子进行单跳问句分类：
```
python Step1_单多跳问句分类.py
```

2. 运行：Step2_单跳问句路径.ipynb，生成候选路径

3. 运行：Step3_生成输入得分计算模型的文件，生成Bert输入文件

4. 计算路径得分：
```
python Step4_3系统路径_score_bert.py
python Step5_模型融合1_ori_ent_rel_score_bert.py
python Step6_模型融合2_nature_en_rel_ori_score_bert.py
python Step7_模型融合3_nature_en_rel_5轮_score_bert.py
python Step8_模型融合3_nature_en_rel_3轮_score_bert.py
```
5. 模型融合，运行：Step9_进行模型的融合.ipynb

6. 生成答案，运行：Step10_找出单跳答案.ipynb


### 链式问句求解
进入文件夹：系统1_模型and规则/路径_答案查找/Step4_链式问句求路径答案/

1. 对剩余问句进行链式问句分类：
```
python Step1_链式问句分类.py
```

2. 生成链式问句和剩余问句候选路径，运行：Step2_找出链式问句中实体的候选path.ipynb

3. 修改输入进BERT的文件格式，运行：Step3_修改模型输入的路径格式.ipynb

4. 计算最优路径：
```
python Step4_链式路径查找_score_bert.py
```
5. 生成链式问句对应答案，运行：Step5_寻找链式答案.ipynb

### 其他问句处理及解决

剩余的问句包括单跳和多跳，由于无法分类，合在一起处理，相关的单跳及多跳路径都生成，这样会导致处理速度会比较慢。

1. 链式问句求解过程中同时生成了该类问句的path，无需重新生成path

2. 对path打分，生成最优路径：
```
python Step4_链式路径查找_score_bert.py
```

3. 生成剩下问句的答案，运行：

#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-06
import sys, os, io
import pickle
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
sys.path.append('../vocab')
sys.path.append('../tools')

def create_seq_labeling_data(qa_data, word_vocab, NoneLabel=0, TrueLabel=1):
    file_type = qa_data.split('.')[-2]
    log_file = open('data/%s.entity_detection.txt' %file_type, 'w')
    pad_index = word_vocab.lookup(word_vocab.pad_token)

    data_list = pickle.load(open(qa_data, 'rb'))
    total=len(data_list)
    print("The length of question set %s is %d"%(file_type,total))

    # 统计最大的问句长度
    max_len_question=0
    for data in data_list:
        max_len_question=max(data.num_text_token,max_len_question)

    EDdata=[]
    for data in data_list:
        if not data.text_attention_indices:
            total-=1
            continue
        tokens = data.question.split()
        len_tokens=data.num_text_token
        labels = data.text_attention_indices
        log_file.write('%s\t%s\n' %(data.question, ' '.join(tokens[labels[0]:labels[-1]+1])))
        seq=torch.LongTensor(max_len_question).fill_(pad_index)
        seq[0:len_tokens]=torch.LongTensor(word_vocab.convert_to_index(tokens))
        seq_label=torch.LongTensor(max_len_question).fill_(NoneLabel)
        seq_label[labels[0]:labels[-1]+1]=TrueLabel
        EDdata.append((seq,seq_label,len_tokens))
    print("The length of question set %s with subject is %d" % (file_type, total))
    torch.save(EDdata, 'data/%s.entity_detection.pt' %file_type)

class SeqLabelingLoader():
    def __init__(self,infile,batch_size,device=-1):
        self.EDdata = torch.load(infile)
        self.batch_size = batch_size
        self.batch_num=math.ceil(len(self.EDdata)/batch_size)

    def gen_batch(self, shuffle = True):
        if shuffle:
            loader=DataLoader(self.EDdata,batch_size=self.batch_size,shuffle=shuffle,collate_fn=self.collate_fn)
        else:
            from ipdb import set_trace
            set_trace()
            loader=DataLoader(self.EDdata,batch_size=self.batch_size,shuffle=False,collate_fn=lambda x:x)
        return loader

    def collate_fn(self,data):
        return sorted(data,key=lambda a:a[-1],reverse=True)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')

    word_vocab = torch.load('../vocab/vocab.word&rel.pt')
    create_seq_labeling_data('../data/QAData.valid.pkl', word_vocab)
    create_seq_labeling_data('../data/QAData.train.pkl', word_vocab)
    create_seq_labeling_data('../data/QAData.test.pkl', word_vocab)

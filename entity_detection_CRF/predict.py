import os
import sys
import numpy as np
import torch
import pickle
import operator
import math

from args import get_args
from model import EntityDetection
from evaluation import evaluation
from seqLabelingLoader import SeqLabelingLoader
sys.path.append('../tools')
import virtuoso
import re

# please set the configuration in the file : args.py
args = get_args()
# set the random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for training")


if not args.trained_model:
    print("ERROR: You need to provide a option 'trained_model' path to load the model.")
    sys.exit(1)

# load word vocab for questions
word_vocab = torch.load(args.vocab_file)
print('load word vocab, size: %s' % len(word_vocab))

os.makedirs(args.results_path, exist_ok=True)

# load the model
#model = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))
model = torch.load(args.trained_model)

def predict(dataset=args.test_file, tp='test', save_qadata=args.save_qadata):
    # load QAdata
    qa_data_path = '../data/QAData.%s.pkl' % tp
    qa_data = pickle.load(open(qa_data_path,'rb'))

    # load batch data for predict
    data_loader = SeqLabelingLoader(dataset, args.batch_size)
    print('load %s data, batch_num: %d\tbatch_size: %d'
            %(tp, data_loader.batch_num, data_loader.batch_size))

    model.eval();

    n_correct = 0
    n_correct_sub = 0
    n_correct_extend = 0
    n_empty = 0
    n_cand_entity=0
    linenum = 1
    qa_data_idx = 0

    new_qa_data = []

    gold_list = []
    pred_list = []
    compare_pred=[]
    
    single_correct=0
    total=0
    EDdata=torch.load(dataset)
  
    batches=[EDdata[i*args.batch_size:(i+1)*args.batch_size] for i in range(math.ceil(len(EDdata)/args.batch_size))]
    for data_batch_idx, data_batch in enumerate(batches):
        if data_batch_idx % 50 == 0:
            print(tp, data_batch_idx)
        seqs,labels,lengths=zip(*data_batch)
        total+=len(lengths)
        # sorted
        seqs,labels,lengths=get_batch_Tensor(seqs,labels,lengths)
        lengths,indices_len=torch.sort(lengths,descending=True)
        seqs=seqs[indices_len]
        labels=labels[indices_len]
        
        scores=model(seqs,lengths)
        mask=model.sequence_mask(lengths,lengths[0])
        
        # recover
        _ , indices_recover=torch.sort(indices_len)
        #from ipdb import set_trace
        #set_trace()
        scores=scores[indices_recover]
        lengths=lengths[indices_recover]
        mask=mask[indices_recover]
        labels=labels[indices_recover]

        paths_batch=model.get_path_topk(scores,mask,topk=args.topk)

        # verify the prediction
        for label,path_topk,length in zip(labels,paths_batch,lengths):
            #subjects_list=predict_subject_name(path_topk)
            #target_subject=predict_subject_name(label)
            for path in path_topk:
                if (path.data==label[:length].data).sum(0)==length:
                    single_correct+=1
        
        for i in range(len(lengths)):
            while qa_data_idx<len(qa_data) and not qa_data[qa_data_idx].text_subject:
                qa_data_idx+=1
            if qa_data_idx>=len(qa_data):
                break
            _qa_data=qa_data[qa_data_idx]
            tokens=_qa_data.question.split()
            # subjects
            predict_sub=predict_subject_ids(paths_batch[i],tokens)
            assert _qa_data.num_text_token==lengths[i]
            if _qa_data.subject in predict_sub:
                n_correct_sub+=1
                #from ipdb import set_trace
                #set_trace()
                '''
                flag=False
                a,b=paths_batch[i].shape
                
                for paths in paths_batch[i]:
                    if (labels[i][:b]==paths).sum()==b:
                        flag=True
                        break
                
                if not flag:
                    print(labels[i][:lengths[i]])
                    print(paths_batch[i])
                    print(_qa_data.subject)
                    print(predict_sub)
                '''
            n_cand_entity+=len(predict_sub)        
            if not predict_sub:
                n_empty+=1
            qa_data_idx+=1
            if save_qadata:
                for sub in predict_sub:
                    rel = virtuoso.id_query_out_rel(sub)
                    _qa_data.add_candidate(sub,rel)
                if hasattr(_qa_data,'cand_rel'):
                    _qa_data.remove_duplicate()
                new_qa_data.append((_qa_data,len(_qa_data.question_pattern.split())))


    print("Average size of candidate entities:%0.6f"%(n_cand_entity/total))
    print("%s\n----------------------------------\n"%(tp))
    name_accuracy=1.0*single_correct/total
    print("name accuracy\taccuracy:%0.6f\tcorrect:%d\ttotal:%d\n"%(name_accuracy,single_correct,total))
    
    id_accuracy=1.0* n_correct_sub/total
    print("id accuracy\taccuracy:%0.6f\tcorrect:%d\ttotal:%d\n"%(id_accuracy,n_correct_sub,total))

    print("subject not found:%0.6f\t%d"%(1.0*n_empty/total,n_empty))
    print("-"*80)
    
    if save_qadata:
        qadata_save_path=open(os.path.join(args.results_path,'QAData.label.%s.pkl'%(tp)),'wb')
        data_list=[data[0] for data in sorted(new_qa_data,key=lambda data:data[1],reverse=True)]
        pickle.dump(data_list,qadata_save_path)
                      
def predict_subject_ids(paths,tokens):
    # single sentence
    predict_subject_ids=[]
    for tags in paths:
        n_subjects=sum(tags[i]==1 and (i-1==-1 or tags[i-1]==0)for i in range(len(tags)))
        if n_subjects==1:
            subject_name=' '.join([tokens[i] for i ,tag in enumerate(tags) if tag==1])
            #start_index=[i for i ,tag in enumerate(tags) if tag==1 and (i==0 or tags[i-1]==0 )][0]
            #end_index=[i for i,tag in enumerate(tags) if tag==1 and(i==len(tags)-1 or tags[i+1]==0)][0]+1
            subject_id=virtuoso.str_query_id(subject_name)
            #subject_id=virtuoso.name_query_id(subject_name)
            predict_subject_ids.extend(subject_id)
    return predict_subject_ids

def get_batch_Tensor(seqs_list,labels_list,lengths_list):
    batch_size=len(seqs_list)
    len_seq=lengths_list[0]
    seqs=torch.LongTensor(batch_size,len_seq)
    labels=torch.LongTensor(batch_size,len_seq)
    lengths=torch.LongTensor(lengths_list)
    for i in range(batch_size):
        seqs[i]=seqs_list[i][:len_seq]
        labels[i]=labels_list[i][:len_seq]
    return seqs,labels,lengths

# run the model on the dev set and write the output to a file
predict(args.valid_file, "valid")

# run the model on the test set and write the output to a file
predict(args.test_file, "test")

# run the model on the train set and write the output to a file
predict(args.train_file, 'train')

import torch
import torch.optim as optim
import torch.nn as nn
import time
import os, sys
import glob
import numpy as np

from args import get_args
from model import EntityDetection
from evaluation import evaluation
from seqLabelingLoader import SeqLabelingLoader

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
    


# load data
train_loader = SeqLabelingLoader(args.train_file,args.batch_size, args.gpu)
print('load train data, batch_num: %d\tbatch_size: %d'
      %(train_loader.batch_num, train_loader.batch_size))
valid_loader = SeqLabelingLoader(args.valid_file,args.batch_size,args.gpu)
print('load valid data, batch_num: %d\tbatch_size: %d'
      %(valid_loader.batch_num, valid_loader.batch_size))

# load word vocab for questions
word_vocab = torch.load(args.vocab_file)
print('load word vocab, size: %s' % len(word_vocab))

os.makedirs(args.save_path, exist_ok=True)

# define models
config = args
config.n_out = 2 # I/in entity  O/out of entity
config.n_cells = config.n_layers

if config.birnn:
    config.n_cells *= 2
print(config)

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = EntityDetection(word_vocab, config)
    if args.word_vectors:
        if os.path.isfile(args.vector_cache):
            pretrained = torch.load(args.vector_cache)
            model.embed.word_lookup_table.weight.data.copy_(pretrained)
        else:
            pretrained = model.embed.load_pretrained_vectors(args.word_vectors, binary=False,
                                            normalize=args.word_normalize)
            torch.save(pretrained, args.vector_cache)
            print('load pretrained word vectors from %s, pretrained size: %s' %(args.word_vectors,
                                                                                pretrained.size()))
   # if args.cuda:
    #    model.cuda()
     #   print("Shift model to GPU")

# show model parameters
for name, param in model.named_parameters():
    print(name, param.size())

criterion = model.crf_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# train the model
iterations = 0
start = time.time()
best_dev_acc = 0
best_dev_F = 0
num_iters_in_epoch = train_loader.batch_num
patience = args.patience * num_iters_in_epoch # for early stopping
iters_not_improved = 0 # this parameter is used for stopping early
early_stop = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss       Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f},{}'.split(','))
best_snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
print(header)

batches=train_loader.gen_batch()
for epoch in range(1, args.epochs+1):
    if early_stop:
        print("Early stopping. Epoch: {}, Best Dev. Acc: {}".format(epoch, best_dev_acc))
        break

    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(batches):
        iterations += 1

        # batch_first
        seqs,labels,lengths=zip(*batch)
        seqs,labels,lengths=get_batch_Tensor(seqs,labels,lengths)

        model.train()
        optimizer.zero_grad()

        scores = model(seqs,lengths)
        
        mask=model.sequence_mask(lengths,lengths[0])
        loss = criterion(scores,mask,labels)
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
        optimizer.step()

        path_batch=model.get_path_topk(scores,mask,topk=args.topk)
        for label,path_topk,length in zip(labels,path_batch,lengths):
            for path in path_topk:
                n_correct += ((path.data == label[:length].data).sum(dim=0)== length).long()
        n_total += len(batch)
        train_acc = 100. * n_correct / n_total

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + \
                        '_iter_{}_acc_{:.4f}_loss_{:.6f}_model.pt'.format(iterations, train_acc, loss.data[0])
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:
            model.eval()
            n_dev_correct = 0
            n_dev_total=0

            gold_list = []
            pred_list = []

            for valid_batch_idx, valid_batch in enumerate(valid_loader.gen_batch()):
                valid_seqs, valid_labels, valid_lengths = zip(*valid_batch)
                valid_seqs,valid_labels,valid_lengths=get_batch_Tensor(valid_seqs, valid_labels, valid_lengths)
                answer = model(valid_seqs,valid_lengths)

                valid_mask = model.sequence_mask(valid_lengths, valid_lengths[0])
                valid_path_batch = model.get_path_topk(answer, valid_mask, topk=args.topk)
                for valid_label, valid_path_topk, valid_length in zip(valid_labels, valid_path_batch, valid_lengths):
                    for valid_path in valid_path_topk:
                        if (valid_path.data == valid_label[:valid_length].data).sum(dim=0) == valid_length:
                            n_dev_correct += 1
                            break
                n_dev_total += len(valid_batch)
            dev_acc = 100. * n_dev_correct / n_dev_total

            print(dev_log_template.format(time.time() - start, epoch, iterations, 
                                          1 + batch_idx, train_loader.batch_num,
                                          100. * (1 + batch_idx) / train_loader.batch_num, 
                                          loss.data[0], train_acc, dev_acc))
            print("{} Precision: {:10.6f}% ".format("Dev", dev_acc))
            # update model
            if dev_acc > best_dev_acc:
                best_dev_acc =dev_acc
                iters_not_improved = 0
                snapshot_path = best_snapshot_prefix + \
                                '_iter_{}_devf1_{}_model.pt'.format(iterations, best_dev_acc)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(best_snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            else:
                iters_not_improved += 1
                if iters_not_improved > patience:
                    early_stop = True
                    break

        # print progress message
        elif iterations % args.log_every == 0:
            print(log_template.format(time.time()-start, epoch, iterations, 1+batch_idx, 
                                      train_loader.batch_num, 100. * (1+batch_idx)/train_loader.batch_num, 
                                      loss.data[0], train_acc, ' '*12))


from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch
import sys
sys.path.append('../tools')
from embedding import Embeddings
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from crf import CRF

class EntityDetection(nn.Module):

    def __init__(self, dicts, config):
        super(EntityDetection, self).__init__()
        self.config = config
        self.embed = Embeddings(word_vec_size=config.d_embed, dicts=dicts)
        if self.config.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=config.d_embed, hidden_size=config.d_hidden,
                              num_layers=config.n_layers, dropout=config.dropout_prob,
                              bidirectional=config.birnn)
        else:
            self.rnn = nn.LSTM(input_size=config.d_embed, hidden_size=config.d_hidden,
                               num_layers=config.n_layers, dropout=config.dropout_prob,
                               bidirectional=config.birnn)

        self.dropout = nn.Dropout(p=config.dropout_prob)
        self.relu = nn.ReLU()
        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2

        self.hidden2tag = nn.Sequential(
                        nn.Linear(seq_in_size, seq_in_size),
                        nn.BatchNorm1d(seq_in_size),
                        self.relu,
                        self.dropout,
                        nn.Linear(seq_in_size, config.n_out)
        )
        self.crf=CRF(config.n_out)

    def forward(self, seqs,length):
        #from ipdb import set_trace
        #set_trace()
        inputs = self.embed.forward(seqs) # shape (batch_size,max_sequence length,dimension of embedding)
        batch_size = inputs.size()[0]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden

        packed=pack_padded_sequence(inputs,length,batch_first=True)
        if self.config.rnn_type.lower() == 'gru':
            h0 = autograd.Variable(inputs.data.new(*state_shape).zero_())
            outputs, ht = self.rnn(packed, h0)
        else:
            h0 = c0 = autograd.Variable(inputs.data.new(*state_shape).zero_())
            outputs, (ht, ct) = self.rnn(packed, (h0, c0))
        outputs,_=pad_packed_sequence(outputs,batch_first=True)
        # shape of `outputs` - (sequence length, batch size, hidden size X num directions)
        tags = self.hidden2tag(outputs.contiguous().view(-1, outputs.size(-1)))
        scores=tags.view(batch_size,-1,tags.size(-1))
        return scores

    def crf_loss(self,scores,mask,labels):
        """
        :param scores: (B,L,dim)
        :param lengths: (B)
        :param labels: (B,L)
        :return:
        """
        loss=self.crf(scores,labels,mask)
        return loss

    def get_path_topk(self,scores,mask,topk=1,batch_first=True):
        if not batch_first:
            scores=torch.transpose(scores,1,0)
        path_batch,_=self.crf.viterbi_tags_batch(scores,mask,topk)
        return path_batch

    def sequence_mask(self,seq_length,max_len=None):
        if not max_len:
            max_len=max(seq_length)
        batch_size=seq_length.size(0)
        seq_range=torch.arange(max_len).long()
        seq_range_expand=seq_range.unsqueeze(0).expand(batch_size,max_len)# B*S
        seq_length_expand=seq_length.unsqueeze(1).expand_as(seq_range_expand)
        return seq_length_expand>seq_range_expand

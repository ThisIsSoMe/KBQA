# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CRF(nn.Module):

    def __init__(self, n_tags):
        super(CRF, self).__init__()

        # 不同的词性个数
        self.n_tags = n_tags
        # 句间迁移(FROM->TO)
        self.trans = nn.Parameter(torch.Tensor(n_tags, n_tags))
        # 句首迁移
        self.strans = nn.Parameter(torch.Tensor(n_tags))
        # 句尾迁移
        self.etrans = nn.Parameter(torch.Tensor(n_tags))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        std = (1 / self.n_tags) ** 0.5
        nn.init.normal_(self.trans, mean=0, std=std)
        nn.init.normal_(self.strans, mean=0, std=std)
        nn.init.normal_(self.etrans, mean=0, std=std)

    def forward(self, emit, target, mask):
        T, B, N = emit.shape

        logZ = self.get_logZ(emit, mask)
        score = self.get_score(emit, target, mask)

        return (logZ - score) / B

    def get_logZ(self, emit, mask):
        T, B, N = emit.shape

        alpha = self.strans + emit[0]  # [B, N]

        for i in range(1, T):
            trans_i = self.trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            mask_i = mask[i].unsqueeze(1).expand_as(alpha)  # [B, N]
            scores = trans_i + emit_i + alpha.unsqueeze(2)  # [B, N, N]
            scores = torch.logsumexp(scores, dim=1)  # [B, N]
            alpha[mask_i] = scores[mask_i]

        return torch.logsumexp(alpha + self.etrans, dim=1).sum()

    def get_score(self, emit, target, mask):
        T, B, N = emit.shape
        scores = torch.zeros(T, B)

        # 加上句间迁移分数
        scores[1:] += self.trans[target[:-1], target[1:]]
        # 加上发射分数
        scores += emit.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)
        # 通过掩码过滤分数
        score = scores.masked_select(mask).sum()

        # 获取序列最后的词性的索引
        ends = mask.sum(dim=0).view(1, -1) - 1
        # 加上句首迁移分数
        score += self.strans[target[0]].sum()
        # 加上句尾迁移分数
        score += self.etrans[target.gather(dim=0, index=ends)].sum()

        return score

    def viterbi(self, emit, mask):
        T, B, N = emit.shape
        lens = mask.sum(dim=0)
        delta = torch.zeros(T, B, N)
        paths = torch.zeros(T, B, N, dtype=torch.long)

        delta[0] = self.strans + emit[0]  # [B, N]

        for i in range(1, T):
            trans_i = self.trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            scores = trans_i + emit_i + delta[i - 1].unsqueeze(2)  # [B, N, N]
            delta[i], paths[i] = torch.max(scores, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            prev = torch.argmax(delta[length - 1, i] + self.etrans)

            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)
            # 反转预测序列并保存
            predicts.append(torch.tensor(predict).flip(0))

        return torch.cat(predicts)

    # FROM: https://gist.github.com/Deepblue129/afaa3613a99a8e7213d2efdd02ae4762
    def viterbi_decode(self,tag_sequence: torch.Tensor, transition_matrix: torch.Tensor, top_k: int = 5):
        """
        Perform Viterbi decoding in log space over a sequence given a transition matrix
        specifying pairwise (transition) potentials between tags and a matrix of shape
        (sequence_length, num_tags) specifying unary potentials for possible tags per
        timestep.
        Parameters
        ----------
        tag_sequence : torch.Tensor, required.
            A tensor of shape (sequence_length, num_tags) representing scores for
            a set of tags over a given sequence.
        transition_matrix : torch.Tensor, required.
            A tensor of shape (num_tags, num_tags) representing the binary potentials
            for transitioning between a given pair of tags.
        top_k : int, required.
            Integer defining the top number of paths to decode.
        Returns
        -------
        viterbi_path : List[int]
            The tag indices of the maximum likelihood tag sequence.
        viterbi_score : float
            The score of the viterbi path.
        """
        sequence_length, num_tags = list(tag_sequence.size())

        path_scores = []
        path_indices = []
        # At the beginning, the maximum number of permutations is 1; therefore, we unsqueeze(0)
        # to allow for 1 permutation.
        path_scores.append(tag_sequence[0, :].unsqueeze(0))
        # assert path_scores[0].size() == (n_permutations, num_tags)

        # Evaluate the scores for all possible paths.
        for timestep in range(1, sequence_length):
            # Add pairwise potentials to current scores.
            # assert path_scores[timestep - 1].size() == (n_permutations, num_tags)
            summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
            summed_potentials = summed_potentials.view(-1, num_tags)

            # Best pairwise potential path score from the previous timestep.
            max_k = min(summed_potentials.size()[0], top_k)
            scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
            # assert scores.size() == (n_permutations, num_tags)
            # assert paths.size() == (n_permutations, num_tags)

            scores = tag_sequence[timestep, :] + scores
            # assert scores.size() == (n_permutations, num_tags)
            path_scores.append(scores)
            path_indices.append(paths.squeeze())

        # Construct the most likely sequence backwards.
        path_scores = path_scores[-1].view(-1)
        max_k = min(path_scores.size()[0], top_k)
        viterbi_scores, best_paths = torch.topk(path_scores, k=max_k, dim=0)
        viterbi_paths = []
        for i in range(max_k):
            viterbi_path = [best_paths[i]]
            for backward_timestep in reversed(path_indices):
                viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
            # Reverse the backward path.
            viterbi_path.reverse()
            # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
            viterbi_path = [j % num_tags for j in viterbi_path]
            viterbi_paths.append(viterbi_path)
        return viterbi_paths, viterbi_scores

    def viterbi_tags(self,logits, mask, top_k):
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        logits: List[List[int]]
        mask: List[int]
        top_k: int
        output: List[List[int]]
        """
        #logits = torch.FloatTensor(logits)
        #from ipdb import set_trace
        #set_trace()
        max_seq_length,num_tags = logits.size()

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        transitions[:num_tags, :num_tags] = self.trans
        transitions[start_tag, :num_tags] = self.strans
        transitions[:num_tags, end_tag] = self.etrans

        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        sequence_length = torch.sum(mask)

        # Start with everything totally unlikely
        tag_sequence.fill_(-10000.)
        # At timestep 0 we must have the START_TAG
        tag_sequence[0, start_tag] = 0.
        # At steps 1, ..., sequence_length we just use the incoming logits
        tag_sequence[1:(sequence_length + 1), :num_tags] = logits[:sequence_length]
        # And at the last timestep we must have the END_TAG
        tag_sequence[sequence_length + 1, end_tag] = 0.

        # We pass the tags and the transitions to ``viterbi_decode``.
        viterbi_paths, viterbi_scores = self.viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions, top_k)
        # Get rid of START and END sentinels and append.
        viterbi_paths = [path[1:-1] for path in viterbi_paths]
        # Ensure that hidden tokens START and END are not in path
        viterbi_paths = [path for path in viterbi_paths if start_tag not in path and end_tag not in path]
        # Translate indexes to labels
        # viterbi_paths = [
        #     [ARCHIVE.model.vocab.get_token_from_index(i, namespace="labels")
        #      for i in paths] for paths in viterbi_paths
        # ]
        return viterbi_paths, viterbi_scores

    def viterbi_tags_batch(self,scores,mask,topk=1):
        """
        :param scores: [B,L,dim]
        :param mask: [B,L]
        :return: path_batch:List[ Tensor[topk,L] ]  score_batch:[ Tensor[topk]]
        """
        B,L,dim=scores.shape
        path_batch=[]
        score_batch=[]
        for i in range(B):
           
            path,score=self.viterbi_tags(scores[i],mask[i],topk)
            path_batch.append(torch.LongTensor(path))
            score_batch.append(torch.FloatTensor(score))
        return path_batch,score_batch

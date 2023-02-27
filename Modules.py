import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import time
import re
from sklearn import preprocessing
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Callable, Optional
from torch import FloatTensor, LongTensor
from typing import List, Any, Iterable
from torch.autograd import Variable

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

class Attn(nn.Module):
    def __init__(self, out_size, attn_size):
        super(Attn, self).__init__()

        self.W = nn.Parameter(torch.Tensor(out_size, out_size))

        self.Wv = nn.Parameter(torch.Tensor(out_size, attn_size))
        self.Wq = nn.Parameter(torch.Tensor(out_size, attn_size))

        nn.init.xavier_uniform_(self.Wv, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.Wq, gain=nn.init.calculate_gain('tanh'))
        
        self.w_hv = nn.Parameter(torch.Tensor(attn_size, 1))
        self.w_hq = nn.Parameter(torch.Tensor(attn_size, 1))
        nn.init.xavier_uniform_(self.w_hv, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.w_hq, gain=nn.init.calculate_gain('linear'))

    def forward(self, seq_features1, seq_features2, mask1, mask2):
        C = torch.tanh(torch.matmul(torch.matmul(seq_features1, self.W), torch.transpose(seq_features2, 1, 2)))

        Hv = torch.tanh(torch.matmul(seq_features1, self.Wv) + torch.matmul(C, torch.matmul(seq_features2, self.Wq)))
        Hq = torch.tanh(torch.matmul(seq_features2, self.Wq) + torch.matmul(torch.transpose(C, 1, 2), torch.matmul(seq_features1, self.Wv)))

        attn_v = masked_softmax(torch.matmul(Hv, self.w_hv).squeeze(), mask1, 1)
        attn_q = masked_softmax(torch.matmul(Hq, self.w_hq).squeeze(), mask2, 1)

        v_hat = torch.sum(torch.unsqueeze(attn_v, 2) * seq_features1, 1)
        q_hat = torch.sum(torch.unsqueeze(attn_q, 2) * seq_features2, 1)
        #print(v_hat.shape,q_hat.shape)
        return v_hat, q_hat
      
class ConstGCN(nn.Module):
  def __init__(
            self,
            hidden_dim,
            w_c_vocab_size,
            c_c_vocab_size,
            non_linearity
    ):
    super(ConstGCN, self).__init__()
    # hidden_dim=768
    self.non_linearity=non_linearity
    # ConstGCN
    # boundary bridging
    self.const_gcn_w_c = ConstGCN_Bridge(
      hidden_dim,
      hidden_dim,
      w_c_vocab_size,
      in_arcs=True,
      out_arcs=True,
      use_gates=True,
      batch_first=True,
      residual=True,
      no_loop=True,
      dropout=0.1,
      non_linearity=self.non_linearity,
      edge_dropout=0.1,
    )

    # reverse boundary bridging
    self.const_gcn_c_w = ConstGCN_Bridge(
      hidden_dim,
      hidden_dim,
      w_c_vocab_size,
      in_arcs=True,
      out_arcs=True,
      use_gates=True,
      batch_first=True,
      residual=True,
      no_loop=True,
      dropout=0.1,
      non_linearity=self.non_linearity,
      edge_dropout=0.1,
    )

    # self graph
    self.const_gcn_c_c = ConstGCN_Bridge(
      hidden_dim,
      hidden_dim,
      c_c_vocab_size,
      in_arcs=True,
      out_arcs=True,
      use_gates=True,
      batch_first=True,
      residual=True,
      no_loop=False,
      dropout=0.1,
      non_linearity=self.non_linearity,
      edge_dropout=0.1,
    )
  def forward(self, const_GCN_w_c,const_GCN_c_w,const_GCN_c_c, const_gcn_in, mask_all):
    # boundary bridging
    adj_arc_in_w_c, adj_arc_out_w_c, adj_lab_in_w_c, adj_lab_out_w_c, mask_in_w_c, mask_out_w_c, mask_loop_w_c = (
        const_GCN_w_c
    )

    # inverse-boundary bridging
    adj_arc_in_c_w, adj_arc_out_c_w, adj_lab_in_c_w, adj_lab_out_c_w, mask_in_c_w, mask_out_c_w, mask_loop_c_w = (
        const_GCN_c_w
    )

    adj_arc_in_c_c, adj_arc_out_c_c, adj_lab_in_c_c, adj_lab_out_c_c, mask_in_c_c, mask_out_c_c, mask_loop_c_c = (
        const_GCN_c_c
    )
    #boundary bridging
    const_gcn_out = self.const_gcn_w_c(
        const_gcn_in,
        adj_arc_in_w_c,
        adj_arc_out_w_c,
        adj_lab_in_w_c,
        adj_lab_out_w_c,
        mask_in_w_c,
        mask_out_w_c,
        mask_loop_w_c,
        mask_all,
    )
    #self-graph
    const_gcn_out = self.const_gcn_c_c(
        const_gcn_out,
        adj_arc_in_c_c,
        adj_arc_out_c_c,
        adj_lab_in_c_c,
        adj_lab_out_c_c,
        mask_in_c_c,
        mask_out_c_c,
        mask_loop_c_c,
        mask_all,
    )
    #print(f'const_gcn_out shape after c_c:{const_gcn_out.shape}')
    const_gcn_out = self.const_gcn_c_w(
        const_gcn_out,
        adj_arc_in_c_w,
        adj_arc_out_c_w,
        adj_lab_in_c_w,
        adj_lab_out_c_w,
        mask_in_c_w,
        mask_out_c_w,
        mask_loop_c_w,
        mask_all,
    )
    return const_gcn_out
class ConstGCN_Bridge(nn.Module):
    """
    Label-aware Constituency Convolutional Neural Network Layer
    """

    def __init__(
            self,
            num_inputs,
            num_units,
            num_labels,
            dropout=0.0,
            in_arcs=True,
            out_arcs=True,
            batch_first=False,
            use_gates=True,
            residual=False,
            no_loop=False,
            non_linearity="relu",
            edge_dropout=0.0,
    ):
        super(ConstGCN_Bridge, self).__init__()

        self.in_arcs = in_arcs
        self.out_arcs = out_arcs
        self.no_loop = no_loop
        self.retain = 1.0 - edge_dropout
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.non_linearity = non_linearity
        self.sigmoid = nn.Sigmoid()
        self.use_gates = use_gates
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(num_units)

        if in_arcs:
            self.V_in = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.V_in)

            self.b_in = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant_(self.b_in, 0)

            if self.use_gates:
                self.V_in_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal_(self.V_in_gate)
                self.b_in_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant_(self.b_in_gate, 1)

        if out_arcs:
            # self.V_out = autograd.Variable(torch.FloatTensor(self.num_inputs, self.num_units))
            self.V_out = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.V_out)

            # self.b_out = autograd.Variable(torch.FloatTensor(num_labels, self.num_units))
            self.b_out = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant_(self.b_out, 0)

            if self.use_gates:
                self.V_out_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal_(self.V_out_gate)
                self.b_out_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant_(self.b_out_gate, 1)
        if not self.no_loop:
            self.W_self_loop = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal_(self.W_self_loop)

            if self.use_gates:
                self.W_self_loop_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal_(self.W_self_loop_gate)

    def forward(
            self,
            src,
            arc_tensor_in=None,
            arc_tensor_out=None,
            label_tensor_in=None,
            label_tensor_out=None,
            mask_in=None,
            mask_out=None,
            mask_loop=None,
            sent_mask=None,
    ):

        if not self.batch_first:
            encoder_outputs = src.permute(1, 0, 2).contiguous()
        else:
            encoder_outputs = src.contiguous()

        batch_size = encoder_outputs.size()[0]
        seq_len = encoder_outputs.size()[1]
        max_degree = 1
        input_ = encoder_outputs.view(
            (batch_size * seq_len, self.num_inputs)
        )  # [b* t, h]
        input_ = self.dropout(input_)
        if self.in_arcs:
            input_in = torch.mm(input_, self.V_in)  # [b* t, h] * [h,h] = [b*t, h]
            first_in = input_in.index_select(
                0, arc_tensor_in[0] * seq_len + arc_tensor_in[1]
            )  # [b* t* degr, h]
            second_in = self.b_in.index_select(0, label_tensor_in[0])  # [b* t* degr, h]
            in_ = first_in + second_in
            degr = int(first_in.size()[0] / batch_size // seq_len)
            in_ = in_.view((batch_size, seq_len, degr, self.num_units))
            if self.use_gates:
                # compute gate weights
                input_in_gate = torch.mm(
                    input_, self.V_in_gate
                )  # [b* t, h] * [h,h] = [b*t, h]
                first_in_gate = input_in_gate.index_select(
                    0, arc_tensor_in[0] * seq_len + arc_tensor_in[1]
                )  # [b* t* mxdeg, h]
                second_in_gate = self.b_in_gate.index_select(0, label_tensor_in[0])
                in_gate = (first_in_gate + second_in_gate).view(
                    (batch_size, seq_len, degr)
                )

            max_degree += degr

        if self.out_arcs:
            input_out = torch.mm(input_, self.V_out)  # [b* t, h] * [h,h] = [b* t, h]
            first_out = input_out.index_select(
                0, arc_tensor_out[0] * seq_len + arc_tensor_out[1]
            )  # [b* t* mxdeg, h]
            second_out = self.b_out.index_select(0, label_tensor_out[0])

            degr = int(first_out.size()[0] / batch_size // seq_len)
            max_degree += degr

            out_ = (first_out + second_out).view(
                (batch_size, seq_len, degr, self.num_units)
            )

            if self.use_gates:
                # compute gate weights
                input_out_gate = torch.mm(
                    input_, self.V_out_gate
                )  # [b* t, h] * [h,h] = [b* t, h]
                first_out_gate = input_out_gate.index_select(
                    0, arc_tensor_out[0] * seq_len + arc_tensor_out[1]
                )  # [b* t* mxdeg, h]
                second_out_gate = self.b_out_gate.index_select(0, label_tensor_out[0])
                out_gate = (first_out_gate + second_out_gate).view(
                    (batch_size, seq_len, degr)
                )
        if self.no_loop:
            if self.in_arcs and self.out_arcs:
                potentials = torch.cat((in_, out_), dim=2)  # [b, t,  mxdeg, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (in_gate, out_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat((mask_in, mask_out), dim=1)  # [b* t, mxdeg]
            elif self.out_arcs:
                potentials = out_  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = out_gate  # [b, t,  mxdeg, h]
                mask_soft = mask_out  # [b* t, mxdeg]
            elif self.in_arcs:
                potentials = in_  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = in_gate  # [b, t,  mxdeg, h]
                mask_soft = mask_in  # [b* t, mxdeg]
            max_degree -= 1
        else:
            same_input = torch.mm(input_, self.W_self_loop).view(
                encoder_outputs.size(0), encoder_outputs.size(1), -1
            )
            same_input = same_input.view(
                encoder_outputs.size(0),
                encoder_outputs.size(1),
                1,
                self.W_self_loop.size(1),
            )
            if self.use_gates:
                same_input_gate = torch.mm(input_, self.W_self_loop_gate).view(
                    encoder_outputs.size(0), encoder_outputs.size(1), -1
                )

            if self.in_arcs and self.out_arcs:
                potentials = torch.cat(
                    (in_, out_, same_input), dim=2
                )  # [b, t,  mxdeg, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (in_gate, out_gate, same_input_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat(
                    (mask_in, mask_out, mask_loop), dim=1
                )  # [b* t, mxdeg]
            elif self.out_arcs:
                potentials = torch.cat(
                    (out_, same_input), dim=2
                )  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (out_gate, same_input_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat((mask_out, mask_loop), dim=1)  # [b* t, mxdeg]
            elif self.in_arcs:
                potentials = torch.cat(
                    (in_, same_input), dim=2
                )  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = torch.cat(
                        (in_gate, same_input_gate), dim=2
                    )  # [b, t,  mxdeg, h]
                mask_soft = torch.cat((mask_in, mask_loop), dim=1)  # [b* t, mxdeg]
            else:
                potentials = same_input  # [b, t,  2*mxdeg+1, h]
                if self.use_gates:
                    potentials_gate = same_input_gate  # [b, t,  mxdeg, h]
                mask_soft = mask_loop  # [b* t, mxdeg]

        potentials_resh = potentials.view(
            (batch_size * seq_len, max_degree, self.num_units)
        )  # [h, b * t, mxdeg]

        if self.use_gates:
            potentials_r = potentials_gate.view(
                (batch_size * seq_len, max_degree)
            )  # [b * t, mxdeg]
            probs_det_ = (self.sigmoid(potentials_r) * mask_soft).unsqueeze(
                2
            )  # [b * t, mxdeg]

            potentials_masked = potentials_resh * probs_det_  # [b * t, mxdeg,h]
        else:
            # NO Gates
            potentials_masked = potentials_resh * mask_soft.unsqueeze(2)

        if self.retain == 1 or not self.training:
            pass
        else:
            mat_1 = torch.Tensor(mask_soft.data.size()).uniform_(0, 1)
            ret = torch.Tensor([self.retain])
            mat_2 = (mat_1 < ret).float()
            drop_mask = Variable(mat_2, requires_grad=False)
            if potentials_resh.is_cuda:
                drop_mask = drop_mask.cuda()

            potentials_masked *= drop_mask.unsqueeze(2)

        potentials_masked_ = potentials_masked.sum(dim=1)  # [b * t, h]

        potentials_masked_ = self.layernorm(potentials_masked_) * sent_mask.view(
            batch_size * seq_len
        ).unsqueeze(1)

        potentials_masked_ = self.non_linearity(potentials_masked_)  # [b * t, h]

        result_ = potentials_masked_.view(
            (batch_size, seq_len, self.num_units)
        )  # [ b, t, h]

        result_ = result_ * sent_mask.unsqueeze(2)  # [b, t, h]
        memory_bank = result_  # [t, b, h]

        if self.residual:
            memory_bank += src

        return memory_bank
class DepGCN(nn.Module):
    """
    Label-aware Dependency Convolutional Neural Network Layer
    """
    def __init__(self, dep_num, dep_dim, in_features, out_features):
        super(DepGCN, self).__init__()
        self.dep_dim = dep_dim
        self.in_features = in_features
        self.out_features = out_features

        self.dep_embedding = nn.Embedding(dep_num, dep_dim, padding_idx=0)
        self.dep_attn = nn.Linear(dep_dim+in_features, out_features)
#         self.dep_attn = nn.Linear(in_features, out_features)
        self.dep_fc = nn.Linear(dep_dim, out_features)
        self.layernorm = nn.LayerNorm(768)
        self.relu = nn.ReLU()

    
    def forward(self, text, dep_mat, dep_labels):

        dep_label_embed = self.dep_embedding(dep_labels) #v_ij

        batch_size, seq_len, feat_dim = text.shape

        val_us = text.unsqueeze(2).repeat_interleave(seq_len, dim=2)
        dep_label_embed_us = dep_label_embed.unsqueeze(2).repeat_interleave(seq_len, dim=2)

        #each output cell in depGCN needs to be a summation of each input node
        val_sum = torch.cat([val_us, dep_label_embed_us], dim=-1) #val_sum= r'_j + v^d_ij = z in paper
        #val_sum=val_us
        r = self.dep_attn(val_sum) # dep_attn = W^d1

#         p = torch.bmm(r, val_us.transpose(2, 3)).squeeze(3) #p=z
#         mask = (dep_mat == 0).float() * (-1e30) #very low values for no arcs relationships
#         p = p + mask 
#         p = torch.softmax(p, dim=2)# no arcs will be very low values
#         p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)
        p = torch.sum(r, dim=-1) #p=z
        mask = (dep_mat == 0).float() * (-1e30) #very low values for no arcs relationships
        p = p + mask 
        p = torch.softmax(p, dim=2)# no arcs will be very low values
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)

        output = val_us + self.dep_fc(dep_label_embed_us) #dep_fc=W^d2
        output = torch.mul(p_us, output)#<- is this where mask is applied?

        output_sum = torch.sum(output, dim=2)
        output_sum = self.relu(output_sum)
        
        return self.layernorm(output_sum)
class Hesyfu(nn.Module):
    def __init__(self, hidden_dim, dep_tag_vocab_size, w_c_vocab_size,
                 c_c_vocab_size, use_constGCN, use_depGCN, device):
        super(Hesyfu, self).__init__()
        self.vocab_size = w_c_vocab_size
        self.dropout = nn.Dropout(p=0.1)
        self.embedding_dropout = nn.Dropout(p=0.2)
        self.use_constGCN = use_constGCN
        self.use_depGCN = use_depGCN
        if self.use_constGCN:
            self.const_gcn = ConstGCN(hidden_dim, w_c_vocab_size,
                                       c_c_vocab_size, nn.ReLU())
        if self.use_depGCN:
            self.dep_gcn = DepGCN(dep_tag_vocab_size, hidden_dim,
                                   hidden_dim, hidden_dim)
        self.gate = nn.Sigmoid()
        self.device = device

    def forward(self, bert_embs1, bert_embs2, data1, data2):

        mask_batch1, lengths_batch1, dependency_arcs1, dependency_labels1, \
        constituent_labels1, const_GCN_w_c1, const_GCN_c_w1, const_GCN_c_c1, \
        mask_const_batch1, plain_sentences1 = data1

        mask_batch2, lengths_batch2, dependency_arcs2, dependency_labels2, \
        constituent_labels2, const_GCN_w_c2, const_GCN_c_w2, const_GCN_c_c2, \
        mask_const_batch2, plain_sentences2 = data2

        # Apply embedding dropout to input embeddings
        base_out1 = self.embedding_dropout(bert_embs1)
        base_out2 = self.embedding_dropout(bert_embs2)
        b1, t1, e1 = base_out1.shape
        b2, t2, e2 = base_out2.shape

        # Apply Constituency GCN
        if self.use_constGCN:
            # Concatenate input embeddings with constituent labels
            const_gcn_in1 = torch.cat([base_out1, constituent_labels1], dim=1)
            const_gcn_in2 = torch.cat([base_out2, constituent_labels2], dim=1)
            
            # Concatenate mask_batch with mask_const_batch
            mask_all1 = torch.cat([mask_batch1, mask_const_batch1], dim=1)
            mask_all2 = torch.cat([mask_batch2, mask_const_batch2], dim=1)
            
            # Apply Constituency GCN
            const_gcn_out1 = self.const_gcn(const_GCN_w_c1, const_GCN_c_w1, const_GCN_c_c1, const_gcn_in1, mask_all1).narrow(1, 0, t1)
            const_gcn_out2 = self.const_gcn(const_GCN_w_c2, const_GCN_c_w2, const_GCN_c_c2, const_gcn_in2, mask_all2).narrow(1, 0, t2)
        
        # Apply Dependency GCN
        if self.use_depGCN:
            # Use Constituency GCN output as input if using it, otherwise use base embeddings
            dep_gcn_in1 = const_gcn_out1 if self.use_constGCN else base_out1
            dep_gcn_in2 = const_gcn_out2 if self.use_constGCN else base_out2
            
            # Apply Dependency GCN
            dep_gcn_out1 = self.dep_gcn(dep_gcn_in1, dependency_arcs1, dependency_labels1)
            dep_gcn_out2 = self.dep_gcn(dep_gcn_in2, dependency_arcs2, dependency_labels2)
            

        # gating
        if self.use_constGCN and self.use_depGCN:
            gate1 = self.gate(dep_gcn_out1 + const_gcn_out1) # changed from concat to elementwise addition
            all_one1 = torch.zeros((b1, t1, gate1.shape[-1]), device=self.device, requires_grad=False)
            hesyfu_out_sentence1 = gate1 * dep_gcn_out1 + (all_one1 - gate1) * const_gcn_out1
            hesyfu_out_sentence1 = self.dropout(hesyfu_out_sentence1)

            gate2 = self.gate(dep_gcn_out2 + const_gcn_out2) # changed from concat to elementwise addition
            all_one2 = torch.zeros((b2, t2, gate2.shape[-1]), device=self.device, requires_grad=False)
            hesyfu_out_sentence2 = gate2 * dep_gcn_out2 + (all_one2 - gate2) * const_gcn_out2
            hesyfu_out_sentence2 = self.dropout(hesyfu_out_sentence2)
            return hesyfu_out_sentence1, hesyfu_out_sentence2
        elif self.use_constGCN:
            return const_gcn_out1, const_gcn_out2
        elif self.use_depGCN:
            return dep_gcn_out1, dep_gcn_out2


    

def flatten_recursive(lst: List[Any]) -> Iterable[Any]:
    """Flatten a list using recursion."""
    for item in lst:
        if isinstance(item, list):
            yield from flatten_recursive(item)
        else:
            yield item
class CA_Hesyfu(nn.Module):
    def __init__(
        self,
        hidden_dim,
        L,
        dep_tag_vocab_size,
        w_c_vocab_size,
        c_c_vocab_size,
        device,
        use_constGCN=True,
        use_depGCN=True
    ):
        super(CA_Hesyfu, self).__init__()
        self.device = device
        self.dep_tag_vocab_size = dep_tag_vocab_size
        self.w_c_vocab_size = w_c_vocab_size
        self.c_c_vocab_size = c_c_vocab_size
        
        self.dropout = nn.Dropout(p=0.1)
        self.embedding_dropout = nn.Dropout(p=0.2)
        
        # BERT
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Hesyfu
        self.hesyfu_layers = nn.ModuleList()
        for _ in range(L):
            hesyfu = Hesyfu(
                hidden_dim,
                dep_tag_vocab_size,
                w_c_vocab_size,
                c_c_vocab_size, 
                use_constGCN,
                use_depGCN,
                self.device
            )
            self.hesyfu_layers.append(hesyfu)
#         self.hesyfu = Hesyfu(
#                     hidden_dim,
#                     dep_tag_vocab_size,
#                     w_c_vocab_size,
#                     c_c_vocab_size, 
#                     use_constGCN,
#                     use_depGCN,
#                     self.device
#                 )
        # Co-attention
        self.co_attn = Attn(768, 768)
        self.fc = nn.Linear(768 * 4, 3)
    def forward(self, sentence1_data, sentence2_data, input_ids, attention_mask, bert_tokenized_sentences):
        # Unpack data
        (mask_batch1, lengths_batch1, dependency_arcs1, dependency_labels1, constituent_labels1,
        const_GCN_w_c1, const_GCN_c_w1, const_GCN_c_c1, mask_const_batch1, plain_sentences1) = sentence1_data
        (mask_batch2, lengths_batch2, dependency_arcs2, dependency_labels2, constituent_labels2,
        const_GCN_w_c2, const_GCN_c_w2, const_GCN_c_c2, mask_const_batch2, plain_sentences2) = sentence2_data
        
        # Pass sentences through BERT
        bert_embs1, bert_embs2 = self.get_bert_output(lengths_batch1, lengths_batch2, plain_sentences1, plain_sentences2,
                                                      input_ids, attention_mask, bert_tokenized_sentences)

        # Pass sentences through GCN's
#         gcn_in1, gcn_in2 = bert_embs1, bert_embs2
        for hesyfu in self.hesyfu_layers:
            gcn_out1, gcn_out2 = hesyfu(gcn_in1, gcn_in2, sentence1_data, sentence2_data)
            gcn_in1, gcn_in2 = gcn_out1, gcn_out2
        
        gcn_out1, gcn_out2 = self.hesyfu(gcn_in1, gcn_in2, sentence1_data, sentence2_data)
        # Pass sentences through co-attention layer
        data1, data2 = self.co_attn(gcn_out1, gcn_out2, mask_batch1, mask_batch2)

        # Create final representation
        final_representation = torch.cat((data1, data2, torch.abs(data1 - data2), torch.mul(data1, data2)), dim=1)
        out = self.fc(final_representation)

        return out
    def _find_corr_features(self, output_features, stanza_tokens, bert_tokens):
        curr_pos = 0
        token_target = ""
        idx_list = []
        feature_tensor = []

        flag = True
        #n_bert_subtokens=0
        #iterate over bert tokens
        for idx, token in enumerate(bert_tokens):
            if token.lower() == stanza_tokens[curr_pos].lower() \
                    or (len(token) == len(stanza_tokens[curr_pos].lower()) and flag == True) \
                    or token == "[UNK]" :
                curr_pos += 1
                idx_list.append([idx])
                token_target = ""
                continue
            
            #token is bert subtoken
            elif token.startswith('##'): #bert subtoken
                #n_bert_subtokens+=1
                token_target += token.lstrip('#')
                idx_list[-1].append(idx)
            else:
                token_target += token
                if flag:
                    idx_list.append([idx])
                    flag = False
                else:
                    idx_list[-1].append([idx])

            if token_target == stanza_tokens[curr_pos].lower() or len(token_target) == len(stanza_tokens[curr_pos].lower()):
                curr_pos += 1
                token_target = ""
                flag = True
        merge_type='mean'
        for sub_idx_list in idx_list:
            flattened_sub_idx_list = list(flatten_recursive(sub_idx_list))
            if merge_type == 'mean':
                sub_feature = torch.mean(output_features[flattened_sub_idx_list[:], :], dim=0, keepdim=False)
            else:
                sub_feature = output_features[flattened_sub_idx_list[0], :]

            sub_feature = sub_feature.unsqueeze(dim=0)
            if len(feature_tensor) > 0:
                feature_tensor = torch.cat((feature_tensor, sub_feature), 0)
            else:
                feature_tensor = sub_feature

        return feature_tensor

    def get_bert_output(self, lengths1, lengths2, plain_sentences1, plain_sentences2, input_ids, attention_mask, bert_tokenized_sentences):
        # Get BERT embeddings for all sentences in the batch
        bert_last_vectors = self.bert(input_ids, attention_mask=attention_mask)[0].to(self.device) 
        batch_size = bert_last_vectors.shape[0]

        # Find the maximum sequence length for each sentence in the batch
        max_seq_len1 = torch.max(lengths1)
        max_seq_len2 = torch.max(lengths2)

        # Initialize empty tensors to store BERT embeddings for each sentence
        bert_embs1 = torch.zeros((batch_size, max_seq_len1, 768)).to(self.device)
        bert_embs2 = torch.zeros((batch_size, max_seq_len2, 768)).to(self.device)

        # Loop over all sentences in the batch and extract BERT embeddings
        for s in range(batch_size):
            # Find the index of the [SEP] token to separate the two input sentences
            sep1_pos = bert_tokenized_sentences[s].index("[SEP]")

            # Extract BERT embeddings for the first sentence
            text1_feature = self._find_corr_features(bert_last_vectors[s, 1:sep1_pos, :], plain_sentences1[s], bert_tokenized_sentences[s][1:sep1_pos])

            # Extract BERT embeddings for the second sentence
            text2_feature = self._find_corr_features(bert_last_vectors[s, sep1_pos+1:-1, :], plain_sentences2[s], bert_tokenized_sentences[s][sep1_pos+1:-1])

            # Find the sequence lengths for the first and second sentences
            seq_len1 = text1_feature.shape[0]
            seq_len2 = text2_feature.shape[0]

            # Store the BERT embeddings in the output tensors
            bert_embs1[s, :seq_len1, :] = text1_feature
            bert_embs2[s, :seq_len2, :] = text2_feature

        return bert_embs1, bert_embs2

def initialize_model(gcn_dim, L, dep_lb_to_idx, w_c_to_idx, c_c_to_idx, device, use_constGCN=True, use_depGCN=True):
    model_name = ''
    if use_constGCN and use_depGCN:
        model_name = 'cahesyfu'
    elif use_constGCN:
        model_name = 'constGCN'
    else:
        model_name = 'depGCN'
    print(f'model name: {model_name}')

    model = CA_Hesyfu(gcn_dim, L, len(dep_lb_to_idx), len(w_c_to_idx), len(c_c_to_idx), device, 
                      use_constGCN=use_constGCN, use_depGCN=use_depGCN)

    count_parameters(model)
    
    return model, model_name
def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total parameters =", num_params)
    return num_params

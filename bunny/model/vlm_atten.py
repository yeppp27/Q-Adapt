# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor





class vlm_cross_attn(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn =  nn.MultiheadAttention(
                          embed_dim=d_model, 
                          kdim=d_model, 
                          vdim=d_model, num_heads=1, batch_first=True)
        
        self.linear1 = nn.Linear(d_model*2, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, 1)
        
    
        self.norm1 = nn.LayerNorm(d_model*2)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
                                    
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        #self.init_zero()
    
    def init_zero(self):
        for param in self.linear1.parameters():
            nn.init.constant_(param,0)
        for param in self.linear2.parameters():
            nn.init.constant_(param,0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     text_emd,
                     vis_emd
                     ):
        q = text_emd
        k = vis_emd  
        v = vis_emd
        src2,attn = self.self_attn(q, k, v)
        src_vl = torch.cat((q,src2),dim=-1)
        src_vl = self.norm1(src_vl)
        alpha = self.linear2(self.dropout(self.activation(self.linear1(src_vl))))
        alpha = nn.Sigmoid()(alpha)
        src3 = (1-alpha)*q+(alpha)*src2  
        src3 = self.norm2(src3)
        return src3,src2,attn

    def forward_pre(self,
                     text_emd,
                     vis_emd
                     ):
        q = k = torch.cat((text_emd,vis_emd),dim=0)
        src2 = self.self_attn(q, k)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src3)
        return src

    def forward(self, text_emd, vis_emd):
        if self.normalize_before:
            return self.forward_pre(text_emd, vis_emd)
        return self.forward_post(text_emd, vis_emd)
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

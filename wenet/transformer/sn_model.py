# Speech and NER model

import torch
import torch.nn as nn

from wenet.transformer.bert import BertEmbedding
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.ctc import CTC

class SNModel(nn.Module):
    '''
    Speech and NER model, fused with cross modality attention.
    speech part use ctc along
    text part use bert
    fuse with one layer cross modality attention
    ---
    bert (string): 
        bert name or path
    bert_out_dim (int): 
        None for the default dim, others add a linear to transform
    bert_n_layers (int): 
        The number of layers from the model to use. If 0, uses all layers.
    bert_dropout (float): 
        The dropout ratio of BERT layers.
    requires_grad (bool):
        If ``True``, the model parameters will be updated together with the downstream task.
        Default: ``True``
    sp_encoder: the speech encoder
    '''
    def __init__(
        self, 
        bert, 
        bert_out_dim, 
        bert_n_layers, 
        bert_dropout,
        sp_encoder: TransformerEncoder,
        ctc: CTC,
        bert_pad_idx=0, 
        bert_requires_grad=True):
        super().__init__()

        self.bert_embed = BertEmbedding(model=bert,
                                        n_layers=bert_n_layers,
                                        n_out=bert_out_dim,
                                        pad_index=bert_pad_idx,
                                        dropout=bert_dropout,
                                        requires_grad=bert_requires_grad)
        
        self.sp_encoder = sp_encoder
        self.ctc = ctc

        



        
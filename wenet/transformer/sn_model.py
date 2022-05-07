# Speech and NER model

import torch
import torch.nn as nn

from wenet.transformer.bert import BertEmbedding
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.ctc import CTC
from wenet.transformer.cma import CrossModalityAttention
from supar.structs import CRFLinearChain
from wenet.utils.mask import make_pad_mask

class BaseNerModel(nn.Module):
    '''
    base ner model just uses text with bert as encoder
    '''
    def __init__(
        self,
        n_labels, 
        bert, 
        bert_out_dim, 
        bert_n_layers, 
        bert_dropout,
        bert_pad_idx=0, 
        bert_requires_grad=True):
        super().__init__()

        self.bert_embed = BertEmbedding(model=bert,
                                        n_layers=bert_n_layers,
                                        n_out=bert_out_dim,
                                        pad_index=bert_pad_idx,
                                        dropout=bert_dropout,
                                        requires_grad=bert_requires_grad)
        self.scorer = nn.Sequential(
                        nn.Linear(bert_out_dim, bert_out_dim//2),
                        nn.ReLU(),
                        nn.Linear(bert_out_dim//2, n_labels)
        )

        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

    def forward(self, words):
        '''
        words: [batch_size, seq_len] plus cls
        '''
        # [batch_size, seq_len, n_out]
        x = self.bert_embed(words.unsqueeze(-1))
        # [batch_size, seq_len, n_labels]
        score = self.scorer(x)
        return score

    def decode(self, score, mask):
        """
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)

    def loss(self, score, gold_labels, mask):
        """
        score: [batch_size, seq_len, n_labels]
        gold_labels: [batch_size, seq_len-1]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        """
        batch_size, seq_len = mask.shape
        loss = -CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans).log_prob(gold_labels).sum() / seq_len
        return loss
        
def init_base_ner_model(configs):
    n_labels = configs['num_ner_labels']
    bert = configs['bert_conf']['bert_path']
    bert_out_dim = configs['bert_conf']['out_dim']
    bert_n_layers = configs['bert_conf']['used_layers']
    bert_dropout = configs['bert_conf']['dropout']
    bert_pad_idx = configs['bert_conf']['pad_idx']
    model = BaseNerModel(n_labels, 
                        bert, 
                        bert_out_dim,
                        bert_n_layers,
                        bert_dropout,
                        bert_pad_idx=bert_pad_idx)
    return model

class SANModel(nn.Module):
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
        n_labels, 
        bert, 
        bert_out_dim, 
        bert_n_layers, 
        bert_dropout,
        sp_encoder: TransformerEncoder,
        ctc: CTC,
        ctc_weight=0.1,
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
        # currently not using masked ctc
        self.ctc = ctc
        self.ctc_weight = ctc_weight
        
        self.cma = CrossModalityAttention(bert_out_dim)

        self.scorer = nn.Sequential(
                        nn.Linear(bert_out_dim, bert_out_dim//2),
                        nn.ReLU(),
                        nn.Linear(bert_out_dim//2, n_labels)
        )
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))

    def forward(self, 
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        bert_tokenid: torch.Tensor):
        '''
        bert_tokenid: [batch_size, seq_len] plus cls
        '''
        # [batch_size, seq_len, d_model]
        bert_repr = self.bert_embed(bert_tokenid.unsqueeze(-1))
        # [batch_size, s_max_len, d_model], [batch_size, 1, s_max_len]
        encoder_out, encoder_mask = self.sp_encoder(speech, speech_lengths)
        s_max_len = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # [batch_size, s_max_len], True is to be padded
        mask = make_pad_mask(encoder_out_lens, s_max_len)
        # [batch_size, seq_len, d_model]
        h = self.cma(bert_repr, encoder_out, encoder_out, mask)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(h)
        return score, encoder_out, encoder_out_lens

    def decode(self, score, mask):
        '''
        score: [batch_size, seq_len, n_labels]
        mask: [batch_size, seq_len]
        seq_len: plus bos(cls)
        '''
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)

    def loss(self, ner_score, gold_ner, ner_mask,
        s_encoder_out,
        s_encoder_lens,
        text: torch.Tensor,
        text_lengths: torch.Tensor):
        batch_size, seq_len = ner_mask.shape
        ner_loss = -CRFLinearChain(ner_score[:, 1:], ner_mask[:, 1:], self.trans).log_prob(gold_ner).sum() / seq_len
        loss_ctc = self.ctc(s_encoder_out, s_encoder_lens, text,
                                text_lengths)
        loss = self.ctc_weight * loss_ctc + (1-self.ctc_weight) * ner_loss
        return loss

    







        



        
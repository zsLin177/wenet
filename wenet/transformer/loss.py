import torch
import torch.nn.functional as F

from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, th_accuracy, add_eos)

class Loss(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ctc_weight: float = 0.5,
        reverse_weight: float = 0.0,
        temp_scalar: int = 1,
        ctc_distill_weight: float = 0.0,
        att_distill_weight: float = 0.0,
        ignore_id: int = IGNORE_ID,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        reduce: bool = True,
        device: torch.device = torch.device("cpu"),
        teacher: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos = vocab_size - 1
        self.device = device
        self.ignore_id = ignore_id
        self.teacher = teacher
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.temp_scalar = temp_scalar
        if self.teacher:
            self.ctc_distill_weight = ctc_distill_weight
            self.att_distill_weight = att_distill_weight
        else:
            self.ctc_distill_weight = 0.0
            self.att_distill_weight = 0.0

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def get_losses(
        self,
        outputs: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor
    ):
        loss_dict = dict()
        loss = torch.zeros(1).to(self.device)
        if self.ctc_weight != 0.0:
            loss_ctc = self._calc_ctc_loss(outputs[0]['ctc_lo_out'],
                                           outputs[0]['encoder_out_lens'],
                                           text, text_lengths)
            loss_dict['loss_ctc'] = loss_ctc
            if len(outputs) == 1:
                loss += loss_ctc * self.ctc_weight
            else:
                loss += loss_ctc * self.ctc_weight * (1 - self.ctc_distill_weight)
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(outputs[0]['decoder_out'],
                                           outputs[0]['r_decoder_out'],
                                           text)
            loss_dict['loss_att'] = loss_att
            if len(outputs) == 1:
                loss += loss_att * (1 - self.ctc_weight)
            else:
                loss += loss_att * (1.0 - self.ctc_weight) * (1 - self.att_distill_weight)

        if self.teacher and len(outputs) > 1:
            if self.ctc_weight != 0.0:
                loss_ctc_distill = self._calc_ce_distilling_loss(
                    outputs[0]['ctc_lo_out'], outputs[1]['ctc_lo_out'])
                loss_dict['loss_ctc_distill'] = loss_ctc_distill
                loss += loss_ctc_distill * self.ctc_weight * self.ctc_distill_weight
            if self.ctc_weight != 1.0:
                loss_decoder_distill = self._calc_kl_distilling_loss(
                    outputs[0]['decoder_out'], outputs[1]['decoder_out'])
                if self.reverse_weight > 0.0:
                    loss_r_decoder_distill = self._calc_kl_distilling_loss(
                        outputs[0]['r_decoder_out'], outputs[1]['r_decoder_out'])
                    loss_decoder_distill = \
                        loss_decoder_distill * (1 - self.reverse_weight) + \
                        loss_r_decoder_distill * self.reverse_weight
                loss_dict['loss_decoder_distill'] = loss_decoder_distill
                loss += loss_decoder_distill * (1 - self.ctc_weight) * self.att_distill_weight

        loss_dict['loss'] = loss
        return loss_dict

    def _calc_ctc_loss(self, encoder_out, encoder_out_lens, text, text_lengths):
        loss = self.ctc_loss(encoder_out, text, encoder_out_lens, text_lengths)
        loss = loss / encoder_out.size(1)
        return loss

    def _calc_att_loss(self, decoder_out, r_decoder_out, ys_pad: torch.Tensor):
        ys_out_pad = add_eos(ys_pad, self.eos, self.ignore_id)
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss, acc_att

    def _calc_ce_distilling_loss(self, y, label):
        loss = F.cross_entropy(y, label.argmax(1))
        return loss

    def _calc_kl_distilling_loss(self, y, label):
        p = F.log_softmax(y/self.temp_scalar, dim=-1)
        q = F.softmax(label/self.temp_scalar, dim=-1)
        loss = F.kl_div(p, q, reduction="none").sum() / y.size(0)
        return loss

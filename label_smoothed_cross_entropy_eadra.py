import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

    # Added coefficients for entropy and distance penalties
    entropy_coef: float = field(
        default=0.0,
        metadata={"help": "coefficient of entropy loss"},
    )
    ave_entropy_coef: float = field(
        default=0.0,
        metadata={"help": "coefficient of average entropy loss"},
    )
    sinkhorn_coef: float = field(
        default=0.0,
        metadata={"help": "coefficient of distance loss"},
    )

    # Additional coefficients for decoder and cross-attention penalties
    dec_entropy_coef: float = field(
        default=0.0,
        metadata={"help": "coefficient of decoder entropy loss"},
    )
    dec_ave_entropy_coef: float = field(
        default=0.0,
        metadata={"help": "coefficient of decoder average entropy loss"},
    )
    dec_sinkhorn_coef: float = field(
        default=0.0,
        metadata={"help": "coefficient of decoder distance loss"},
    )
    decx_entropy_coef: float = field(
        default=0.0,
        metadata={"help": "coefficient of cross-attention entropy loss"},
    )
    decx_ave_entropy_coef: float = field(
        default=0.0,
        metadata={"help": "coefficient of cross-attention average entropy loss"},
    )
    decx_sinkhorn_coef: float = field(
        default=0.0,
        metadata={"help": "coefficient of cross-attention distance loss"},
    )

    entropy_head: int = field(
        default=0,
        metadata={"help": "Number of heads to reduce entropy"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """
    Compute the label-smoothed negative log-likelihood loss.

    Args:
        lprobs (Tensor): Log probabilities of the model predictions.
        target (Tensor): Ground-truth target tensor.
        epsilon (float): Smoothing factor.
        ignore_index (int, optional): Index to ignore in the target.
        reduce (bool, optional): If True, reduces the loss to a scalar.

    Returns:
        Tuple[Tensor, Tensor]: Total loss and negative log-likelihood loss.
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        entropy_coef,
        entropy_head,
        sinkhorn_coef,
        ave_entropy_coef,
        dec_entropy_coef,
        dec_sinkhorn_coef,
        dec_ave_entropy_coef,
        decx_entropy_coef,
        decx_sinkhorn_coef,
        decx_ave_entropy_coef,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        # Added entropy and distance penalty coefficients
        self.entropy_coef = entropy_coef
        self.entropy_head = entropy_head
        self.sinkhorn_coef = sinkhorn_coef
        self.ave_entropy_coef = ave_entropy_coef

        # Decoder-specific penalty coefficients
        self.dec_entropy_coef = dec_entropy_coef
        self.dec_sinkhorn_coef = dec_sinkhorn_coef
        self.dec_ave_entropy_coef = dec_ave_entropy_coef

        # Cross-attention penalty coefficients
        self.decx_entropy_coef = decx_entropy_coef
        self.decx_sinkhorn_coef = decx_sinkhorn_coef
        self.decx_ave_entropy_coef = decx_ave_entropy_coef

    def entropy_penalty(self, attention_distributions):
        """
        Compute the entropy penalty for attention distributions.

        Added by the user to encourage diversity across attention distributions.
        """
        eps = 1e-8  # Small positive value to prevent log(0)
        attention_distributions = attention_distributions.clamp(min=eps)
        entropy = -torch.sum(
            attention_distributions * torch.log2(attention_distributions), dim=(-2, -1)
        )
        normalized_entropy = entropy / (
            math.log2(attention_distributions.shape[-1])
            * attention_distributions.shape[-2]
        )
        column_averages = torch.mean(attention_distributions, dim=-2)
        column_entropy = -torch.sum(
            column_averages * torch.log2(column_averages), dim=-1
        )
        normalized_column_entropy = column_entropy / math.log2(
            column_averages.shape[-1]
        )
        return normalized_entropy.sum(), normalized_column_entropy.sum()

    def distance_penalty(self, attention_distributions):
        """
        Compute the distance penalty for attention distributions.

        Added by the user to prioritize attending to adjacent tokens.
        """
        seq_len = attention_distributions.shape[-1]
        distance_matrix = torch.abs(
            torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
        ).float().to(attention_distributions.device)
        total_distance = torch.bmm(
            attention_distributions[:, :, :, :-1].view(-1, 1, seq_len - 1),
            (distance_matrix @ attention_distributions[:, :, :, 1:].transpose(-1, -2))
            .view(-1, seq_len - 1, 1),
        ).sum()
        return total_distance

    def compute_loss(self, model, net_output, encoder_out, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        # Extract attention distributions
        enc_out = torch.stack(encoder_out["selfattn"])
        dec_selfattn = torch.stack(net_output[1]["decoder_selfattn"])
        dec_xattn = torch.stack(net_output[1]["decoder_xattn"])

        # Compute entropy and distance penalties for encoder
        enc_entropy, enc_ave_entropy = self.entropy_penalty(enc_out)
        enc_distance = self.distance_penalty(enc_out)

        # Compute penalties for decoder self-attention
        dec_entropy, dec_ave_entropy = self.entropy_penalty(dec_selfattn)
        dec_distance = self.distance_penalty(dec_selfattn)

        # Compute penalties for cross-attention
        decx_entropy, decx_ave_entropy = self.entropy_penalty(dec_xattn)
        decx_distance = self.distance_penalty(dec_xattn)

        # Combine all penalties
        total_loss = (
            loss
            + self.entropy_coef * enc_entropy
            - self.ave_entropy_coef * enc_ave_entropy
            + self.sinkhorn_coef * enc_distance
            + self.dec_entropy_coef * dec_entropy
            - self.dec_ave_entropy_coef * dec_ave_entropy
            + self.dec_sinkhorn_coef * dec_distance
            + self.decx_entropy_coef * decx_entropy
            - self.decx_ave_entropy_coef * decx_ave_entropy
            + self.decx_sinkhorn_coef * decx_distance
        )

        return (
            total_loss,
            nll_loss,
            enc_distance,
            enc_entropy,
            enc_ave_entropy,
        )

    def forward(self, model, sample, reduce=True):
        net_output, encoder_out = model(**sample["net_input"])
        loss, nll_loss, sinkhorn_dist, entropy, entropy_avg = self.compute_loss(
            model, net_output, encoder_out, sample, reduce=reduce
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "sum_sink_dist": sinkhorn_dist,
            "entropy": entropy,
            "entropy_average": entropy_avg,
        }
        return loss, sample_size, logging_output


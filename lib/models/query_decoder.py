from torch import nn
from transformers import BertModel
import torch
import math


class GroupWiseLinear(nn.Module):
    """
    Taken directly from https://github.com/SlongLiu/query2labels/blob/main/lib/models/query2label.py
    """

    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Bert_with_Decoder(nn.Module):

    def __init__(self, num_labels: int = 4, d_model: int = 2048, num_decoder: int = 2,
                 n_heads: int = 4, frozen_bert=False, bert_type='bert-base-uncased',
                 dropout=0.1, decoder_ffn_dim=8192):

        """
        Query2Label framework as described in
        https://arxiv.org/abs/2107.10834

        Attributes
        ----------
        """

        super().__init__()

        """
        Backbone side
        """
        self.backbone = BertModel.from_pretrained(bert_type)

        # Freeze BERT parameters if specified
        if frozen_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get Backbone last dimension size
        self.d_backbone = list(self.backbone.children())[-1].state_dict()['dense.weight'].shape[1]

        self.lin1 = nn.Linear(self.d_backbone, d_model)
        """
        Decoder side
        """
        # Init label embedding
        # ref: https://github.com/SlongLiu/query2labels/blob/main/lib/models/query2label.py, line 68
        self.label_embedding_weights = nn.Parameter(torch.zeros(1, num_labels, d_model))

        # Init decoders
        # note for norm = None, refer Issue: https://github.com/pytorch/pytorch/issues/24930
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads,
                                                   dim_feedforward=decoder_ffn_dim,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder, norm=None)
        # TODO: rebuild TransformerDecoder to output intemediaries and attention matrix
        # TODO: consider positional embedding (not implemented)
        self.dropout = nn.Dropout(dropout)
        self.fc = GroupWiseLinear(num_labels, d_model, bias=True)

        self._reset_param()

        self.d_model = d_model
        self.n_heads = n_heads

    def _reset_param(self):

        # for p in self.parameters():
        #     if p.dim() > 1:  # excludes Biases (no symmetry issues)
        #         nn.init.xavier_uniform_(p)

        self.label_embedding_weights = torch.nn.init.normal_(self.label_embedding_weights)

    def forward(self, input_ids, attn_mask):

        batch_size = input_ids.shape[0]

        """
        Backbone pass
        """
        bert_out = self.backbone(input_ids, attn_mask)[0]  # (N, max_seq_len, d_backbone)

        # Linear projection to transform to input size required by decoders
        memory = self.lin1(bert_out)

        """
        Decoder pass
        """
        # Repeat label embeddings to go from (num_classes, d_model) to (N, num_classes, d_model)
        query_label = self.label_embedding_weights.repeat(batch_size, 1, 1)
        decoder_out = self.decoder(query_label, memory)  # (N, num_classes, d_model)
        dropout_out = self.dropout(decoder_out)
        logits = self.fc(dropout_out)

        return logits

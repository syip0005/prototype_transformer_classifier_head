from torch import nn
from transformers import BertModel


class Bert_Simple_Classifier(nn.Module):

    def __init__(self, dropout: float = 0.1, num_labels: int = 2, frozen_bert=True, bert_type='bert-base-uncased'):

        """
        Base BERT model to be used as baseline model.

        Attributes
        ----------
        dropout : int
            probability of dropout for classification layer
        num_labels : int
            dimension of classifier output
        frozen_bert : bool
            determine whether parameters of BERT are fine-tuned,
            otherwise just tune the classifier parameters
        bert_type : str
            argument for .from_pretrained()
        """

        super().__init__()

        # Backbone
        self.backbone = BertModel.from_pretrained(bert_type)

        # Freeze BERT parameters if specified
        if frozen_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get Backbone last dimension size
        dim = list(self.backbone.children())[-1].state_dict()['dense.weight'].shape[1]

        # Final Classification Layer
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(dim, num_labels)

    def forward(self, input_id, attn_mask):

        # Returns just [CLS] encoding
        backbone_out = self.backbone(input_id, attn_mask)[1]
        drop_out = self.dropout(backbone_out)
        logits = self.fc(drop_out)

        return logits

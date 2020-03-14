import torch.nn as nn
from transformers import BertModel
import torch

class SentimentClassifier(nn.Module):
    def __init__(self, num_classes, device, freeze_bert = True):
        super(SentimentClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.device = device

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.cls_layer = nn.Linear(768, num_classes)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        #Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits.to(self.device)

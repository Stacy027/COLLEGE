import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForMaskedLM
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pretrained = RobertaForMaskedLM.from_pretrained('roberta-base')
        self.pretrained.roberta.encoder.gradient_checkpointing = True

        self.config = self.pretrained.config
        self.pretrained.to(self.device)
        
        self.text_projection = nn.Parameter(torch.empty(self.config.hidden_size, self.config.hidden_size))
        self.init_parameters()

    def init_parameters(self):

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.config.hidden_size ** -0.5)

    def forward(self, text):
        # n_word_nodes = text['n_word_nodes'][0]
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
        token_type_ids = text['token_type_ids']
        position_ids = text['position_ids']
        masked_lm_labels = text['masked_lm_labels']

        out = self.pretrained.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids
                    )
        # print(out.last_hidden_state[:, 0])
        sequence_output = out[0]
        output = out.last_hidden_state[:, 0] @ self.text_projection
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        word_logits = self.pretrained.lm_head(sequence_output)

        word_predict = torch.argmax(word_logits, dim=-1)
        masked_lm_loss = loss_fct(word_logits.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        # out = self.fc(out.last_hidden_state[:, 0])
        # out = out.softmax(dim=1)

        return output, masked_lm_loss

class GraphEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        self.pretrained = RobertaForMaskedLM.from_pretrained('roberta-base')
        self.pretrained.roberta.encoder.gradient_checkpointing = True
        self.config = self.pretrained.config
        
        self.pretrained.to(self.device)
        
        self.graph_projection = nn.Parameter(torch.empty(self.config.hidden_size, self.config.hidden_size))
        self.init_parameters()

    def extend_type_embedding(self, token_type=3):
        self.pretrained.roberta.embeddings.token_type_embeddings = nn.Embedding(token_type, self.config.hidden_size,
                                                                     _weight=torch.zeros(
                                                                         (token_type, self.config.hidden_size)))
    

    def init_parameters(self):
        if self.graph_projection is not None:
            nn.init.normal_(self.graph_projection, std=self.config.hidden_size ** -0.5)

    def forward(self, graph):
        # n_word_nodes = graph['n_word_nodes'][0]
        input_ids = graph['input_ids']
        attention_mask = graph['attention_mask']
        token_type_ids = graph['token_type_ids']
        position_ids = graph['position_ids']
        masked_lm_labels = graph['masked_lm_labels']

        out = self.pretrained.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids
                    )
        # print(out.last_hidden_state[:, 0])
        sequence_output = out[0]
        output =  out.last_hidden_state[:, 0] @ self.graph_projection
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        word_logits = self.pretrained.lm_head(sequence_output)
        word_predict = torch.argmax(word_logits, dim=-1)
        masked_lm_loss = loss_fct(word_logits.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        return output, masked_lm_loss
    

class Clip(nn.Module):

    def __init__(self, device, alpha=0.6):
        super().__init__()
        self.alpha = alpha
        self.device = device
        self.textbert = TextEncoder(device)
        self.graphbert = GraphEncoder(device)
        self.graphbert.extend_type_embedding(token_type=3)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    
    def forward(self,text, graph):
        text_features, mlm_loss_text = self.textbert(text)
        graph_features, mlm_loss_ent = self.graphbert(graph)
        text_features = F.normalize(text_features, dim=-1)
        graph_features = F.normalize(graph_features, dim=-1)

        logits_per_graph = torch.matmul(text_features, graph_features.T) * self.logit_scale.exp()
        logits_per_text = torch.matmul(graph_features, text_features.T) * self.logit_scale.exp()

        labels = torch.arange(logits_per_graph.shape[0], dtype=torch.long).to(self.device)
        
        loss = (
            F.cross_entropy(logits_per_graph, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        # print(logits_per_text)
        total_loss = self.alpha * loss + (1-self.alpha)/2 * mlm_loss_text + (1-self.alpha)/2 * mlm_loss_ent

        return total_loss, loss, mlm_loss_text, mlm_loss_ent, logits_per_text
    
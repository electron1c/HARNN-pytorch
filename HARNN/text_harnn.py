# -*- coding:utf-8 -*-
__author__ = 'hy'

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class AttentionLayer(nn.Module):
    def __init__(self, num_units, attention_unit_size, num_classes):
        super(AttentionLayer, self).__init__()
        self.fc1 = nn.Linear(num_units, attention_unit_size, bias=False)
        self.fc2 = nn.Linear(attention_unit_size, num_classes, bias=False)

    def forward(self, input_x):
        attention_matrix = self.fc2(torch.tanh(self.fc1(input_x))).transpose(1, 2)
        attention_weight = torch.softmax(attention_matrix, dim=-1)
        attention_out = torch.matmul(attention_weight, input_x)
        return attention_weight, torch.mean(attention_out, dim=1)


class LocalLayer(nn.Module):
    def __init__(self, num_units, num_classes):
        super(LocalLayer, self).__init__()
        self.fc = nn.Linear(num_units, num_classes)

    def forward(self, input_x, input_att_weight):
        logits = self.fc(input_x)
        scores = torch.sigmoid(logits)
        visual = torch.mul(input_att_weight, scores.unsqueeze(-1))
        visual = torch.softmax(visual, dim=-1)
        visual = torch.mean(visual, dim=1)
        return logits, scores, visual


class TextHARNN(nn.Module):
    """A HARNN for text classification."""

    def __init__(
            self, num_classes_list, total_classes, vocab_size, lstm_hidden_size,
            attention_unit_size, fc_hidden_size, embedding_size, embedding_type, beta=0.0,
            pretrained_embedding=None, dropout_keep_prob=None):
        super(TextHARNN, self).__init__()
        self.beta = beta

        # Embedding Layer
        if pretrained_embedding is None:
            embedding_weight = torch.FloatTensor(np.random.uniform(-1, 1, size=(vocab_size, embedding_size)))
            embedding_weight = Variable(embedding_weight, requires_grad=True)
        else:
            embedding_weight = torch.from_numpy(pretrained_embedding)
            if embedding_type == 1:
                embedding_weight = Variable(embedding_weight, requires_grad=True)
        self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=embedding_weight)

        # Bi-LSTM Layer
        self.bi_lstm = nn.LSTM(input_size=embedding_size, hidden_size=lstm_hidden_size, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=dropout_keep_prob)

        # First Level
        self.first_attention = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[0])
        self.first_fc = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.first_local = LocalLayer(fc_hidden_size, num_classes_list[0])

        # Second Level
        self.second_attention = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[1])
        self.second_fc = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.second_local = LocalLayer(fc_hidden_size, num_classes_list[1])

        # Third Level
        self.third_attention = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[2])
        self.third_fc = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.third_local = LocalLayer(fc_hidden_size, num_classes_list[2])

        # Fourth Level
        self.fourth_attention = AttentionLayer(lstm_hidden_size * 2, attention_unit_size, num_classes_list[3])
        self.fourth_fc = nn.Linear(lstm_hidden_size * 4, fc_hidden_size)
        self.fourth_local = LocalLayer(fc_hidden_size, num_classes_list[3])

        # Fully Connected Layer
        self.fc = nn.Linear(fc_hidden_size * 4, fc_hidden_size)

        # Highway Layer
        self.highway_lin = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.highway_gate = nn.Linear(fc_hidden_size, fc_hidden_size)

        # Add dropout
        self.dropout = nn.Dropout(dropout_keep_prob)

        # Global scores
        self.global_scores_fc = nn.Linear(fc_hidden_size, total_classes)

    def forward(self, input_x):
        # Embedding Layer
        embedded_sentence = self.embedding(input_x)
        # Average Vectors
        # [batch_size, embedding_size]
        embedded_sentence_average = torch.mean(embedded_sentence, dim=1)

        # Bi-LSTM Layer
        lstm_out, _ = self.bi_lstm(embedded_sentence)
        lstm_out_pool = torch.mean(lstm_out, dim=1)

        # First Level
        first_att_weight, first_att_out = self.first_attention(lstm_out)
        first_local_input = torch.cat((lstm_out_pool, first_att_out), dim=1)
        first_local_fc_out = self.first_fc(first_local_input)
        first_logits, first_scores, first_visual = self.first_local(first_local_fc_out, first_att_weight)

        # Second Level
        second_att_input = torch.mul(lstm_out, first_visual.unsqueeze(-1))
        second_att_weight, second_att_out = self.second_attention(second_att_input)
        second_local_input = torch.cat((lstm_out_pool, second_att_out), dim=1)
        second_local_fc_out = self.second_fc(second_local_input)
        second_logits, second_scores, second_visual = self.second_local(second_local_fc_out, second_att_weight)

        # Third Level
        third_att_input = torch.mul(lstm_out, second_visual.unsqueeze(-1))
        third_att_weight, third_att_out = self.third_attention(third_att_input)
        third_local_input = torch.cat((lstm_out_pool, third_att_out), dim=1)
        third_local_fc_out = self.third_fc(third_local_input)
        third_logits, third_scores, third_visual = self.third_local(third_local_fc_out, third_att_weight)

        # Fourth Level
        fourth_att_input = torch.mul(lstm_out, third_visual.unsqueeze(-1))
        fourth_att_weight, fourth_att_out = self.fourth_attention(fourth_att_input)
        fourth_local_input = torch.cat((lstm_out_pool, fourth_att_out), dim=1)
        fourth_local_fc_out = self.second_fc(fourth_local_input)
        fourth_logits, fourth_scores, fourth_visual = self.fourth_local(fourth_local_fc_out, fourth_att_weight)

        # Concat
        # shape of ham_out: [batch_size, fc_hidden_size * 4]
        ham_out = torch.cat((first_local_fc_out, second_local_fc_out,
                             third_local_fc_out, fourth_local_fc_out), dim=1)

        # Fully Connected Layer
        fc_out = self.fc(ham_out)

        # Highway Layer and Dropout
        highway_g = torch.relu(self.highway_lin(fc_out))
        highway_t = torch.sigmoid(self.highway_gate(fc_out))
        highway_output = torch.mul(highway_g, highway_t) + torch.mul((1 - highway_t), fc_out)
        h_drop = self.dropout(highway_output)

        # Global scores
        global_logits = self.global_scores_fc(h_drop)
        global_scores = torch.sigmoid(global_logits)
        local_scores = torch.cat((first_scores, second_scores, third_scores, fourth_scores), dim=1)
        scores = self.beta * global_scores + (1 - self.beta) * local_scores
        return scores, (first_logits, second_logits, third_logits, fourth_logits,
                        global_logits, first_scores, second_scores)


if __name__ == '__main__':
    textHARNN = TextHARNN(num_classes_list=[9, 128, 661, 8364], total_classes=9162,
                          vocab_size=1024, lstm_hidden_size=256, attention_unit_size=200, fc_hidden_size=512,
                          embedding_size=100, embedding_type=1, beta=0.5, dropout_keep_prob=0.5)
    test_input = torch.LongTensor([[0, 0, 0]])
    test_output = textHARNN(test_input)
    print(test_output)


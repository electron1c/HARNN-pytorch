# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import torch
import torch.nn as nn
from text_harnn import TextHARNN
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import torch.optim
from utils import data_helper as dh
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Parameters
# train_or_restore = input("☛ Train or Restore?(T/R): ")
# while not (train_or_restore.isalpha() and train_or_restore.upper() in ['T', 'R']):
#     train_or_restore = input("✘ The format of your input is illegal, please re-input: ")
# train_or_restore = train_or_restore.upper()

# Data Parameters
training_data_file = '../data/train_sample.json'
validation_data_file = '../data/validation_sample.json'

# Hyper parameters
learning_rate = 0.001
pad_seq_len = 150
embedding_dim = 100
embedding_type = 1
lstm_hidden_size = 256
attention_unit_size = 200
attention_penalization = True
fc_hidden_size = 512
dropout_keep_prob = 0.5
l2_reg_lambda = 0
alpha = 0
beta = 0.5
num_classes_list = [9, 128, 661, 8364]
total_classes = 9162
top_num = 5
threshold = 0.5
harnn_type = 'TextHARNN'

# Training Parameters
batch_size = 4
num_epochs = 5


class Loss(nn.Module):
    def __init__(self, alpha):
        super(Loss, self).__init__()
        self.alpha = alpha
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.MSELoss = nn.MSELoss()

    def forward(self, first_logits, second_logits, third_logits, fourth_logits,
                global_logits, first_scores, second_scores, input_y_first,
                input_y_second, input_y_third, input_y_fourth, input_y):
        # Local Loss
        losses_1 = self.BCEWithLogitsLoss(first_logits, input_y_first.float())
        losses_2 = self.BCEWithLogitsLoss(second_logits, input_y_second.float())
        losses_3 = self.BCEWithLogitsLoss(third_logits, input_y_third.float())
        losses_4 = self.BCEWithLogitsLoss(fourth_logits, input_y_fourth.float())
        local_losses = losses_1 + losses_2 + losses_3 + losses_4

        # Global Loss
        global_losses = self.BCEWithLogitsLoss(global_logits, input_y.float())

        # Hierarchical violation Loss
        if alpha == 0:
            return local_losses + global_losses
        index_list = [(0, 15), (15, 37), (52, 20), (72, 9), (81, 7), (88, 17), (106, 14), (120, 5), (125, 3)]
        violation_losses = 0.0
        for i in range(len(index_list)):
            (left_index, step) = index_list[i]
            current_parent_scores = first_scores[:, i]
            current_child_scores = second_scores[:, left_index:left_index+step]
            margin = torch.max(current_child_scores - current_parent_scores, 0)
            losses = self.MSELoss(margin)
            violation_losses = violation_losses + self.alpha * losses

        return local_losses + global_losses + violation_losses


def train_harnn():
    writer = SummaryWriter('runs/harnn_experiment_1')
    print("Loading Data")
    train_data = dh.load_data_and_labels(training_data_file, num_classes_list,
                                         total_classes, embedding_dim, data_aug_flag=False)
    val_data = dh.load_data_and_labels(validation_data_file, num_classes_list,
                                       total_classes, embedding_dim, data_aug_flag=False)
    x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3 = dh.pad_data(train_data, pad_seq_len)
    x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3 = dh.pad_data(val_data, pad_seq_len)
    train_dataset = TensorDataset(x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3)
    val_dataset = TensorDataset(x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    vocab_size, pretrained_word2vec_matrix = dh.load_word2vec_matrix(embedding_dim)

    print("Init nn")
    net = TextHARNN(num_classes_list=num_classes_list, total_classes=total_classes,
                    vocab_size=vocab_size, lstm_hidden_size=lstm_hidden_size, attention_unit_size=attention_unit_size,
                    fc_hidden_size=fc_hidden_size, embedding_size=embedding_dim, embedding_type=embedding_type,
                    beta=beta, dropout_keep_prob=dropout_keep_prob)
    criterion = Loss(alpha)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_reg_lambda)

    print("Training")
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        for x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3 in train_loader:
            optimizer.zero_grad()
            _, outputs = net(x_train)
            loss = criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
                             y_train_0, y_train_1, y_train_2, y_train_3, y_train)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            writer.add_scalar('training loss', running_loss / 1000, epoch * len(train_loader) + i)
            i += 1
    writer.add_graph(net, x_train)
    writer.close()

    print('Finished Training')


if __name__ == "__main__":
    train_harnn()


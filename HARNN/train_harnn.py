# -*- coding:utf-8 -*-
__author__ = 'hy'

import time
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import torch.optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from text_harnn import TextHARNN
from utils import data_helper as dh

# Parameters
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
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
batch_size = 4
num_epochs = 61
evaluate_every = 5
checkpoint_every = 10
best_auprc = 0


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


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
        # index_list = [(0, 15), (15, 37), (52, 20), (72, 9), (81, 7), (88, 17), (106, 14), (120, 5), (125, 3)]
        # violation_losses = 0.0
        # for i in range(len(index_list)):
        #     (left_index, step) = index_list[i]
        #     current_parent_scores = first_scores[:, i]
        #     current_child_scores = second_scores[:, left_index:left_index+step]
        #     margin = torch.max(current_child_scores - current_parent_scores, 0)
        #     losses = self.MSELoss(margin)
        #     violation_losses = violation_losses + self.alpha * losses
        #
        # return local_losses + global_losses + violation_losses


def train_harnn():
    global best_auprc
    train_or_restore = input("☛ Train or Restore?(T/R): ")
    while not (train_or_restore.isalpha() and train_or_restore.upper() in ['T', 'R']):
        train_or_restore = input("✘ The format of your input is illegal, please re-input: ")
    train_or_restore = train_or_restore.upper()
    if train_or_restore == 'T':
        logger = dh.logger_fn("training", "log/training-{0}.log".format(str(int(time.time()))))
    else:
        logger = dh.logger_fn("training", "log/restore-{0}.log".format(str(int(time.time()))))

    logger.info("Loading Data...")
    train_data = dh.load_data_and_labels(training_data_file, num_classes_list,
                                         total_classes, embedding_dim, data_aug_flag=False)
    val_data = dh.load_data_and_labels(validation_data_file, num_classes_list,
                                       total_classes, embedding_dim, data_aug_flag=False)
    x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3 = dh.pad_data(train_data, pad_seq_len)
    x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3 = dh.pad_data(val_data, pad_seq_len)
    train_dataset = TensorDataset(x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3)
    val_dataset = TensorDataset(x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    vocab_size, pretrained_word2vec_matrix = dh.load_word2vec_matrix(embedding_dim)

    logger.info("Init nn...")
    net = TextHARNN(num_classes_list=num_classes_list, total_classes=total_classes,
                    vocab_size=vocab_size, lstm_hidden_size=lstm_hidden_size, attention_unit_size=attention_unit_size,
                    fc_hidden_size=fc_hidden_size, embedding_size=embedding_dim, embedding_type=embedding_type,
                    beta=beta, pretrained_embedding=pretrained_word2vec_matrix, dropout_keep_prob=dropout_keep_prob).to(device)
    criterion = Loss(alpha)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_reg_lambda)
    if train_or_restore == 'R':
        model = input("☛ Please input the checkpoints model you want to restore: ")
        while not (model.isdigit() and len(model) == 10):
            model = input("✘ The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs/model", model))
        checkpoint = torch.load(out_dir)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        net.train()
        best_auprc = checkpoint['best_auprc'].to(device)

    logger.info("Training...")
    # writer = SummaryWriter('summary')
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_cnt = 0
        for x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3 in train_loader:
            x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3 = \
                [i.to(device) for i in [x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3]]
            optimizer.zero_grad()
            _, outputs = net(x_train)
            loss = criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
                             y_train_0, y_train_1, y_train_2, y_train_3, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_cnt += x_train.size()[0]
            logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, train_cnt + 1, train_loss / train_cnt))
            # writer.add_scalar('training loss', train_loss / train_cnt, epoch * len(train_dataset) + train_cnt)
        if epoch % evaluate_every == 0:
            val_loss = 0.0
            val_cnt = 0
            eval_pre_tk = [0.0 for _ in range(top_num)]
            eval_rec_tk = [0.0 for _ in range(top_num)]
            eval_F_tk = [0.0 for _ in range(top_num)]
            true_onehot_labels = []
            predicted_onehot_scores = []
            predicted_onehot_labels_ts = []
            predicted_onehot_labels_tk = [[] for _ in range(top_num)]
            for x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3 in val_loader:
                x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3 = \
                    [i.to(device) for i in [x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3]]
                scores, outputs = net(x_val)
                scores = scores[0]
                loss = criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
                                 y_val_0, y_val_1, y_val_2, y_val_3, y_val)
                val_loss += loss.item()
                val_cnt += x_val.size()[0]
                # Prepare for calculating metrics
                for onehot_labels in y_val:
                    true_onehot_labels.append(onehot_labels.tolist())
                for onehot_scores in scores:
                    predicted_onehot_scores.append(onehot_scores.tolist())
                # Predict by threshold
                batch_predicted_onehot_labels_ts = \
                    dh.get_onehot_label_threshold(scores=scores.cpu().detach().numpy(), threshold=threshold)
                for onehot_labels in batch_predicted_onehot_labels_ts:
                    predicted_onehot_labels_ts.append(onehot_labels)
                # Predict by topK
                for num in range(top_num):
                    batch_predicted_onehot_labels_tk = \
                        dh.get_onehot_label_topk(scores=scores.cpu().detach().numpy(), top_num=num + 1)
                    for onehot_labels in batch_predicted_onehot_labels_tk:
                        predicted_onehot_labels_tk[num].append(onehot_labels)

            # Calculate Precision & Recall & F1
            eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                          y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                       y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            eval_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                                 y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            # Calculate the average AUC
            eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                     y_score=np.array(predicted_onehot_scores), average='micro')
            # Calculate the average PR
            eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                               y_score=np.array(predicted_onehot_scores), average='micro')
            is_best = eval_prc > best_auprc
            best_auprc = max(eval_prc, best_auprc)

            for num in range(top_num):
                eval_pre_tk[num] = precision_score(y_true=np.array(true_onehot_labels),
                                                   y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')
                eval_rec_tk[num] = recall_score(y_true=np.array(true_onehot_labels),
                                                y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')
                eval_F_tk[num] = f1_score(y_true=np.array(true_onehot_labels),
                                          y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')
            logger.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                        .format(val_loss / val_cnt, eval_auc, eval_prc))
            logger.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}"
                        .format(eval_pre_ts, eval_rec_ts, eval_F_ts))
            logger.info("Predict by topK:")
            for num in range(top_num):
                logger.info("Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}"
                            .format(num + 1, eval_pre_tk[num], eval_rec_tk[num], eval_F_tk[num]))
            # writer.add_scalar('validation loss', val_loss / val_cnt, epoch)
            # writer.add_scalar('validation AUC', eval_auc, epoch)
            # writer.add_scalar('validation AUPRC', eval_prc, epoch)

        if epoch % checkpoint_every == 0:
            timestamp = str(int(time.time()))
            save_checkpoint({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auprc': best_auprc,
            }, is_best, filename=os.path.join(os.path.curdir, "model", "epoch%d.%s.pth" % (epoch, timestamp)))
    # writer.add_graph(net, x_train)
    # writer.close()

    logger.info('Finished Training.')


if __name__ == "__main__":
    train_harnn()


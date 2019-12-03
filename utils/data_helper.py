# -*- coding:utf-8 -*-
__author__ = 'ljq'

import heapq
import multiprocessing
import logging
import gensim
import json
import os
import torch
import numpy as np
from collections import OrderedDict
from gensim.models import word2vec


TEXT_DIR = '../data/content.txt'
METADATA_DIR = '../data/metadata.tsv'


def logger_fn(name, input_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def create_prediction_file(output_file, data_id, all_labels, all_predict_labels, all_predict_scores):
    """
    Create the prediction file.
    Args:
        output_file: The all classes predicted results provided by network
        data_id: The data record id info provided by class Data
        all_labels: The all origin labels
        all_predict_labels: The all predict labels by threshold
        all_predict_scores: The all predict scores by threshold
    Raises:
        IOError: If the prediction file is not a <.json> file
    """
    if not output_file.endswith('.json'):
        raise IOError("✘ The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(all_predict_labels)
        for i in range(data_size):
            predict_labels = [int(i) for i in all_predict_labels[i]]
            predict_scores = [round(i, 4) for i in all_predict_scores[i]]
            labels = [int(i) for i in all_labels[i]]
            data_record = OrderedDict([
                ('id', data_id[i]),
                ('labels', labels),
                ('predict_labels', predict_labels),
                ('predict_scores', predict_scores)
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted onehot labels based on the topK number.
    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_label_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_labels: The predicted labels
        predicted_scores: The predicted scores
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        score_list = []
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                index_list.append(index)
                score_list.append(predict_score)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            score_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def get_label_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK number.
    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def create_metadata_file(embedding_size, output_file=METADATA_DIR):
    """
    Create the metadata file based on the corpus file(Use for the Embedding Visualization later).
    Args:
        embedding_size: The embedding size
        output_file: The metadata file (default: 'metadata.tsv')
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist."
                      "Please use function <create_vocab_size(embedding_size)> to create it!")

    model = gensim.models.Word2Vec.load(word2vec_file)
    word2idx = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def create_word2vec_model(embedding_size, input_file=TEXT_DIR):
    """
    Create the word2vec model based on the given embedding size and the corpus file.
    Args:
        embedding_size: The embedding size
        input_file: The corpus file
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    sentences = word2vec.LineSentence(input_file)
    # sg=0 means use CBOW model(default); sg=1 means use skip-gram model.
    model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
                                   sg=0, workers=multiprocessing.cpu_count())
    model.save(word2vec_file)


def load_word2vec_matrix(embedding_size):
    """
    Return the word2vec model matrix.
    Args:
        embedding_size: The embedding size
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist. "
                      "Please use function <create_vocab_size(embedding_size)> to create it!")
    model = gensim.models.Word2Vec.load(word2vec_file)
    vocab_size = len(model.wv.vocab.items())
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    vector = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():
        if key is not None:
            vector[value] = model[key]
    return vocab_size, vector


def data_word2vec(input_file, num_classes_list, total_classes, word2vec_model):
    """
    Create the research data tokenindex based on the word2vec model file.
    Return the class Data(includes the data tokenindex and data labels).
    Args:
        input_file: The research data
        num_classes_list: <list> The number of classes
        total_classes: The total number of classes
        word2vec_model: The word2vec model file
    Returns:
        The class Data(includes the data tokenindex and data labels)
    Raises:
        IOError: If the input file is not the .json file
    """
    # num_classes_list = list(map(int, num_classes_list.split(',')))
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def _token_to_index(content):
        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    def _create_onehot_labels(labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label

    if not input_file.endswith('.json'):
        raise IOError("✘ The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    with open(input_file) as fin:
        id_list = []
        title_index_list = []
        abstract_index_list = []
        labels_list = []
        onehot_labels_list = []
        onehot_labels_tuple_list = []
        total_line = 0

        for eachline in fin:
            data = json.loads(eachline)
            patent_id = data['id']
            title_content = data['title']
            abstract_content = data['abstract']
            first_labels = data['section']
            second_labels = data['subsection']
            third_labels = data['group']
            fourth_labels = data['subgroup']
            total_labels = data['labels']

            id_list.append(patent_id)
            title_index_list.append(_token_to_index(title_content))
            abstract_index_list.append(_token_to_index(abstract_content))
            labels_list.append(total_labels)
            labels_tuple = (_create_onehot_labels(first_labels, num_classes_list[0]),
                            _create_onehot_labels(second_labels, num_classes_list[1]),
                            _create_onehot_labels(third_labels, num_classes_list[2]),
                            _create_onehot_labels(fourth_labels, num_classes_list[3]))

            onehot_labels_tuple_list.append(labels_tuple)
            onehot_labels_list.append(_create_onehot_labels(total_labels, total_classes))
            total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def patent_id(self):
            return id_list

        @property
        def title_tokenindex(self):
            return title_index_list

        @property
        def abstract_tokenindex(self):
            return abstract_index_list

        @property
        def labels(self):
            return labels_list

        @property
        def onehot_labels_tuple(self):
            return onehot_labels_tuple_list

        @property
        def onehot_labels(self):
            return onehot_labels_list

    return _Data()


def data_augmented(data, drop_rate=1.0):
    """
    Data augmented.
    Args:
        data: The Class Data()
        drop_rate: The drop rate
    Returns:
        aug_data
    """
    aug_num = data.number
    aug_patent_id = data.patent_id
    aug_title_tokenindex = data.title_tokenindex
    aug_abstract_tokenindex = data.abstract_tokenindex
    aug_labels = data.labels
    aug_onehot_labels = data.onehot_labels
    aug_onehot_labels_tuple = data.onehot_labels_tuple

    for i in range(len(data.aug_abstract_tokenindex)):
        data_record = data.tokenindex[i]
        if len(data_record) == 1:
            continue
        elif len(data_record) == 2:
            data_record[0], data_record[1] = data_record[1], data_record[0]
            aug_patent_id.append(data.patent_id[i])
            aug_title_tokenindex.append(data.title_tokenindex[i])
            aug_abstract_tokenindex.append(data_record)
            aug_labels.append(data.labels[i])
            aug_onehot_labels.append(data.onehot_labels[i])
            aug_onehot_labels_tuple.append(data.onehot_labels_tuple[i])
            aug_num += 1
        else:
            data_record = np.array(data_record)
            for num in range(len(data_record) // 10):
                # random shuffle & random drop
                data_shuffled = np.random.permutation(np.arange(int(len(data_record) * drop_rate)))
                new_data_record = data_record[data_shuffled]

                aug_patent_id.append(data.patent_id[i])
                aug_title_tokenindex.append(data.title_tokenindex[i])
                aug_abstract_tokenindex.append(list(new_data_record))
                aug_labels.append(data.labels[i])
                aug_onehot_labels.append(data.onehot_labels[i])
                aug_onehot_labels_tuple.append(data.onehot_labels_tuple[i])
                aug_num += 1

    class _AugData:
        def __init__(self):
            pass

        @property
        def number(self):
            return aug_num

        @property
        def patent_id(self):
            return aug_patent_id

        @property
        def title_tokenindex(self):
            return aug_title_tokenindex

        @property
        def abstract_tokenindex(self):
            return aug_abstract_tokenindex

        @property
        def labels(self):
            return aug_labels

        @property
        def onehot_labels(self):
            return aug_onehot_labels

        @property
        def onehot_labels_tuple(self):
            return aug_onehot_labels_tuple

    return _AugData()


def load_data_and_labels(data_file, num_classes_list, total_classes, embedding_size, data_aug_flag):
    """
    Load research data from files, splits the data into words and generates labels.
    Return split sentences, labels and the max sentence length of the research data.
    Args:
        data_file: The research data
        num_classes_list: <list> The number of classes
        total_classes: The total number of classes
        embedding_size: The embedding size
        data_aug_flag: The flag of data augmented
    Returns:
        The class Data
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    # Load word2vec model file
    if not os.path.isfile(word2vec_file):
        create_word2vec_model(embedding_size, TEXT_DIR)

    model = word2vec.Word2Vec.load(word2vec_file)

    # Load data from files and split by words
    data = data_word2vec(data_file, num_classes_list, total_classes, word2vec_model=model)
    if data_aug_flag:
        data = data_augmented(data)

    # plot_seq_len(data_file, data)

    return data


def pad_sequence_with_maxlen(sequences, batch_first=False, padding_value=0, maxlen_arg=None):
    r"""
    Change from the raw code in torch.nn.utils.rnn for the need to pad with a assigned length
    Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
        maxlen:the the max length you want to pad

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    # if maxlen_arg != None and maxlen_arg < max_len:
    #   max_len = max_len_arg
    if maxlen_arg == None:
        max_len = max([s.size(0) for s in sequences])
    else:
        max_len = maxlen_arg
    #

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = min(max_len, tensor.size(0))
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor[:length]
        else:
            out_tensor[:length, i, ...] = tensor[:length]

    return out_tensor


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Return the padded data and data labels.
    Args:
        data: The research data
        pad_seq_len: The max sentence length of research data
    Returns:
        pad_seq: The padded data
        labels: The data labels
    """
    abstract_pad_seq = pad_sequence_with_maxlen([torch.tensor(item) for item in data.abstract_tokenindex],
                                                batch_first=True, padding_value=0., maxlen_arg=pad_seq_len)
    # abstract_pad_seq = abstract_pad_seq.numpy()
    # abstract_pad_seq = pad_sequences(data.abstract_tokenindex, maxlen=pad_seq_len, value=0.)

    onehot_labels_list = data.onehot_labels
    onehot_labels_list_tuple = data.onehot_labels_tuple
    return abstract_pad_seq, torch.tensor(onehot_labels_list), \
           torch.tensor(np.array(onehot_labels_list_tuple)[:, 0].tolist()), \
           torch.tensor(np.array(onehot_labels_list_tuple)[:, 1].tolist()), \
           torch.tensor(np.array(onehot_labels_list_tuple)[:, 2].tolist()), \
           torch.tensor(np.array(onehot_labels_list_tuple)[:, 3].tolist())


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。
    Args:
        data: The data
        batch_size: The size of the data batch
        num_epochs: The number of epochs
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

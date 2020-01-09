import os
import numpy as np
from sklearn.model_selection import train_test_split


def get_standard_file(config):
    """
    :param config: get train data and test data file path
    :return: None, generate two new files: train_data.txt and test_data.txt
    """

    # read standard file to get char_col and tag_col
    char_col, tag_col = [], []
    with open(config.source, "r", encoding='utf8', errors='ignore') as f:

        for line in f.readlines():
            if line == '\n':
                continue

            # store char of word and correspond tag
            seq_char, seq_tag = [], []

            # get word list by split space and skip the first word(time)
            word_list = list(filter(None, line.strip('\n').split(' ')))[1:]

            for index in range(len(word_list)):
                # get chinese word
                word = word_list[index].split("/")[0]

                # skip special symbol "["
                if word[0] == "[":
                    word = word[1:]

                # get char of word and correspond tag
                # length of word is 1 denote "S"
                if len(word) == 1:
                    seq_char.append(word)
                    seq_tag.append("S")

                else:
                    for i, char in enumerate(word):
                        seq_char.append(char)

                        # the first of word tag B
                        if i == 0:
                            seq_tag.append("B")

                        elif i == len(word) - 1:
                            seq_tag.append("E")

                        else:
                            seq_tag.append("M")

            char_col.append(seq_char)
            tag_col.append(seq_tag)

    char_col = np.array(char_col)
    tag_col = np.array(tag_col)

    train_data, test_data, train_tag, test_tag = train_test_split(char_col, tag_col, test_size=0.25, random_state=42)

    # write to train and test data
    write_file(train_data, train_tag, config.train)
    write_file(test_data, test_tag, config.test)


# write data and tag to file path
def write_file(data, tag, path):
    """
    :param data: char collection
    :param tag: corresponding tag collection
    :param path: the write file path
    :return: None, Only write data and tag to file
    """

    write_str = ""
    for i in range(len(data)):
        for j in range(len(data[i])):
            write_str += data[i][j] + "\t" + tag[i][j] + "\n"
        write_str += "\n"

    with open(path, "w", encoding='utf8', errors='ignore') as f:
        f.write(write_str)


# get vocab to create emission matrix
def get_vocab(path, vocab, index, maxLen):
    """
    :param path: the data file path
    :param vocab: init vocab
    :param index: init index
    :param maxLen: init max length of sequence
    :return: vocab with the path of file data and max length
    """
    with open(path, "r", encoding='utf8', errors='ignore') as f:
        sindex = 0

        for i, line in enumerate(f.readlines()):
            if line == '\n':
                maxLen = max(i - sindex, maxLen)
                sindex = i + 1
                continue

            word = line.split('\t')[0]
            if word not in vocab:
                vocab[word] = index
                index += 1
    return vocab, index, maxLen


# get train or test data and target
def get_data(path, vocab, class_dict, maxLen):
    data, target = [], []

    with open(path, "r", encoding='utf8', errors='ignore') as f:
        seq_data, seq_target = [], []
        for line in f.readlines():

            if line == "\n":
                data.append(seq_data + [0] * (maxLen - len(seq_data)))
                target.append(seq_target + [-1] * (maxLen - len(seq_target)))
                seq_data, seq_target = [], []
                continue

            temp_list = list(filter(None, line.strip('\n').split('\t')))
            char, tag = temp_list[0], temp_list[1]

            seq_data.append(vocab[temp_list[0]])
            seq_target.append(class_dict[tag])

    return data, target

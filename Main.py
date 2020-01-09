import os
import numpy
import argparse

import DataProcess
import Evaluation
import Config
import Model
import Data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def TrainModel(model, optimizer, train_loader, epoch, config):
    epoch_loss = 0
    for index, (data, target) in enumerate(train_loader):
        data = torch.tensor(data).to(config.device)
        target = torch.tensor(target).to(config.device)

        output = model(data)
        output = output.view(-1, config.num_classes)
        target = target.view(-1)

        loss = F.cross_entropy(output, target, ignore_index=-1)
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch is {} and loss is {:.2f}".format(epoch, epoch_loss))


def TestModel(model, test_loader, config, res_eval):
    with torch.no_grad():
        for index, (data, target) in enumerate(test_loader):
            data = torch.tensor(data).to(config.device)
            target = torch.tensor(target).to(config.device)

            output = torch.argmax(model(data), dim=2)
            # output = torch.argmax(output.view(-1, config.num_classes))
            print(target.size(), output.size())
            # target = target.view(-1)

            res_eval.add(target, output)

    res_eval.eval_model()


def Terminal_parser():
    # define default train and test data
    source_path = "./data/people-daily.txt"
    train_path = "./data/train_data.txt"
    test_path = "./data/test_data.txt"

    parser = argparse.ArgumentParser()
    parser.description = "choose some parameters with terminal"

    parser.add_argument("--source", help='the path of source data', default=source_path)
    parser.add_argument("--train", help='the path of train data', default=train_path)
    parser.add_argument("--test", help='the path of test data', default=test_path)

    parser.add_argument("--input_size", help="input dimension", default=200)
    parser.add_argument("--hidden_size", help="model hidden dimension", default=200)
    parser.add_argument("--num_classes", help='the number of class', default=4)
    parser.add_argument("--num_layers", help="the number of layers", default=2)
    parser.add_argument("--batch_size", help="data batch size", default=256)
    parser.add_argument("--epoch", help="train and test epoch", default=10)
    parser.add_argument("--device", help="device type", default="cpu")
    # parser.add_argument("--device", help="device type", default="cuda")

    args = parser.parse_args()
    return args


def main():
    # using Terminal to get parameters
    config = Config.Config_Table(Terminal_parser())

    # get standard train and test file
    if not os.path.exists(config.train) or not os.path.exists(config.test):
        DataProcess.get_standard_file(config)

    # get char vocab
    vocab, index, maxLen = {}, 5, -1
    vocab, index, maxLen = DataProcess.get_vocab(config.train, vocab, index, maxLen)
    vocab, _, maxLen = DataProcess.get_vocab(config.test, vocab, index, maxLen)
    config.n_feature = len(vocab)

    # get standard train and test data
    train_data, train_target = DataProcess.get_data(config.train, vocab, config.class_dict, maxLen)
    test_data, test_target = DataProcess.get_data(config.test, vocab, config.class_dict, maxLen)

    # get train loader and test loader
    train_loader = Data.get_dataloader(train_data, train_target, config.batch_size)
    test_loader = Data.get_dataloader(test_data, test_target, config.batch_size)

    # define model and optimizer
    model = Model.LSTM_Linear(config, vocab).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Train and Test Model")
    res_eval = Evaluation.Result_Eval(config.num_classes)
    # train and test model
    for epoch in range(config.epochs):
        TrainModel(model, optimizer, train_loader, epoch, config)
        TestModel(model, test_loader, config, res_eval)
    res_eval.best_model_result()


if __name__ == "__main__":
    main()

import torch


class Result_Eval(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y = None
        self.y_hat = None
        self.flag = 1

        self.opt_precision = 0
        self.opt_recall = 0
        self.opt_F_measure = 0

    # add each batch size of y and y_hat
    def add(self, y, y_hat):
        if self.flag == 1:
            self.flag = 0
            self.y, self.y_hat = y, y_hat
        else:
            self.y = torch.cat((self.y, y), dim=0)
            self.y_hat = torch.cat((self.y_hat, y_hat), dim=0)

    # get each sequence word as {start_index: word_length}
    def get_dict(self, y, maskLen):
        word_dict, s_index, TruthLen = {}, -1, -1

        for i in range(maskLen):
            # "s" means a token is a word
            if y[i] == 0:
                word_dict[i] = 1

            # "B" means a word begin
            elif y[i] == 1:
                s_index = i

            # "E" means end of word
            elif y[i] == 3:
                if s_index != -1:
                    word_dict[s_index] = (i - s_index) + 1
                s_index = -1

            # -1 means ignore
            elif y[i] == -1:
                TruthLen = i
                break

        return word_dict, TruthLen

    def eval_model(self):
        self.flag = 1
        TruthNum, PredictionNum, CorrectNum = 0, 0, 0

        for i in range(self.y.size(0)):
            maskLen = self.y[i].size(0)
            y_dict, maskLen = self.get_dict(self.y[i], maskLen)
            y_hat_dict, _ = self.get_dict(self.y_hat[i], maskLen)

            TruthNum += len(y_dict)
            PredictionNum += len(y_hat_dict)
            for key, value in y_hat_dict.items():
                if key in y_dict and y_dict[key] == value:
                    CorrectNum += 1

        Precision = CorrectNum/PredictionNum * 100
        Recall = CorrectNum/TruthNum * 100
        F_measure = 2 * Precision * Recall / (Precision + Recall)
        print("Precision Value: {:.2f}".format(Precision))
        print("Recall Value: {:.2f}".format(Recall))
        print("F-Measure Value: {:.2f}".format(F_measure))

        if self.opt_F_measure < F_measure:
            self.opt_F_measure = F_measure
            self.opt_precision = Precision
            self.opt_recall = Recall

    def best_model_result(self):
        print()
        print("===================================================")
        print("Best Model Evaluation")
        print("===================================================")
        print("Precision Value: {:.2f}".format(self.opt_precision))
        print("Recall Value: {:.2f}".format(self.opt_recall))
        print("F-Measure Value: {:.2f}".format(self.opt_F_measure))
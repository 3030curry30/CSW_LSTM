class Config_Table(object):
    def __init__(self, args):
        self.n_classes = args.num_classes
        self.source = args.source
        self.train = args.train
        self.test = args.test

        self.n_feature = 0
        self.class_dict = {"S": 0, "B": 1, "M": 2, "E": 3}
        self.invert_class_dict = {0: "S", 1: "B", 2: "M", 3: "E"}

        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.epochs = args.epoch
        self.device = args.device

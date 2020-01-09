from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]


def collate_fn(col):
    data = list(list(zip(*col))[0])
    target = list(list(zip(*col))[1])
    return data, target


def get_dataloader(data, target, batch_size):
    data_set = MyData(data, target)
    data_loader = DataLoader(data_set, batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader

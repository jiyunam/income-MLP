import torch.utils.data as data

class AdultDataset(data.Dataset):
    def __init__(self, features, labels):
        # 3.1 YOUR CODE HERE
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # 3.1 YOUR CODE HERE
        features = self.features[index]
        label = self.labels[index]
        return features, label

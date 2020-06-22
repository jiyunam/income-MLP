import torch.nn as nn

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size):
        super(MultiLayerPerceptron, self).__init__()

        # 3.3 YOUR CODE HERE
        self.input_size = input_size
        self.output_size = 1

        self.fc1 = nn.Linear(self.input_size, 64)                       # comment out for 4.4
        # self.fc1 = nn.Linear(self.input_size, self.output_size)       # use for 4.4
        self.act1 = nn.Tanh()                                           # comment out for 4.6
        # self.act1 = nn.ReLU()                                         # use for 4.6
        # self.act1 = nn.Sigmoid()                                      # use for 4.6
        self.fc2 = nn.Linear(64, self.output_size)                      # comment out for 4.4, 4.5
        # self.fc2 = nn.Linear(64, 64)                                  # use for 4.5
        self.act2 = nn.Sigmoid()                                        # comment out for 4.5

        # self.act2 = nn.Tanh()                                         # use for 4.5
        # self.fc3 = nn.Linear(64, 64)                                  # use for 4.5
        # self.act3 = nn.Tanh()                                         # use for 4.5
        # self.fc4 = nn.Linear(64, self.output_size)                    # use for 4.5
        # self.act4 = nn.Sigmoid()                                      # use for 4.5


    def forward(self, features):
        # 3.3 YOUR CODE HERE
        x = self.fc1(features)
        x = self.act1(x)
        x = self.fc2(x)                                             # comment out for 4.4
        x = self.act2(x)

        # x = self.fc3(x)                                           # use for 4.5
        # x = self.act3(x)                                          # use for 4.5
        # x = self.fc4(x)                                           # use for 4.5
        # x = self.act4(x)                                          # use for 4.5
        return x
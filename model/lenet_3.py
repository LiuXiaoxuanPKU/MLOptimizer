# '''LeNet in PyTorch.'''
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# # class LeNet(nn.Module):
# #     def __init__(self):
# #         super(LeNet, self).__init__()
# #         self.features = nn.Sequential(
# #             nn.Conv2d(3, 6, 5),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(2),
# #
# #             nn.Conv2d(6, 16, 5),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(2),
# #         )
# #
# #         self.classifier = nn.Sequential(
# #             nn.Linear(16 * 5 * 5, 120),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(120, 84),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(84, 10),
# #         )
# #
# #     def forward(self, x):
# #         x = self.features(x)
# #         x = torch.flatten(x, 1)
# #         x = self.classifier(x)
# #         return x

'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.fc1   = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3   = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
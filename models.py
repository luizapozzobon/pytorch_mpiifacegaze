import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        model = models.alexnet(pretrained=True)
        self.alexnet = model.features

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Sequential(
            nn.Linear(256*13*13, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(4096, 2)

        self._initialize_weight(mode="normal")
        self._initialize_bias()

    def _initialize_weight(self, mode='xavier'):
        if mode == 'xavier':
            nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu'))
        else:
            nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.conv3.weight, mean=0.0, std=0.001)

    def _initialize_bias(self):
        nn.init.constant_(self.conv1.bias, val=0.1)
        nn.init.constant_(self.conv2.bias, val=0.1)
        nn.init.constant_(self.conv3.bias, val=1)

    def forward(self, x):
        x = self.alexnet(x)

        # Spatial weights
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        w = F.relu(self.conv3(y))

        # Element wise multiplication of alexnet output with spatial weights
        x = F.dropout(F.relu(torch.mul(x, w)), 0.5)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x

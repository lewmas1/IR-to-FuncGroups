import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 1000)
        x = self.linear_layers(x)
        return x
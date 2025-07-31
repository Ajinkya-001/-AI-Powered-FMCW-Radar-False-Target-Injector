import torch.nn as nn

class FalseTargetGenerator(nn.Module):
    def __init__(self):
        super(FalseTargetGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.net(x)

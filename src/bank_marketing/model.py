import torch.nn as nn


class BankMarketingMLP(nn.Module):
    def __init__(self, input_dim, dropout1=.5, dropout2=.4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(.01), nn.Dropout(dropout1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(.01), nn.Dropout(dropout2),
            nn.Linear(64, 1))
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a=.01, nonlinearity='leaky_relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)


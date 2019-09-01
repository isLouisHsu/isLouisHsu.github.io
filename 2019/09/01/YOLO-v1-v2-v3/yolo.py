import torch
from torch import nn

def _maxpool2x2():
    return nn.MaxPool2d(2, 2)

def _conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )

def _conv1x1(in_channels, out_channels):
    return _conv(in_channels, out_channels, 1, 1, 0)

def _conv3x3(in_channels, out_channels):
    return _conv(in_channels, out_channels, 3, 1, 1)

def _subConv(in_channels, out_channels):
    return nn.Sequential(
        _conv1x1(in_channels, out_channels // 2),
        _conv3x3(out_channels // 2, out_channels) 
    )

class YOLOv1(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()

        self.S = S; self.B = B; self.C = C

        self.features = self._make_layer()
        self.conn = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C))
        )

    def _make_layer(self):

        cfg = [256, 512, 'M', 512, 512, 512, 512, 1024, 'M', 1024, 1024]

        layers = [
            _conv(3, 64, 7, 2, 1),
            _maxpool2x2(),
            _conv(64, 192, 3, 1, 1),
            _maxpool2x2(),
        ]

        in_channels = 192
        for c in cfg:
            if c == 'M':
                layers += [_maxpool2x2()]
            else:
                layers += [_subConv(in_channels, c)]
                in_channels = c

        layers += [
            _conv3x3(1024, 1024),
            _conv   (1024, 1024, 3, 2, 1),
            _conv3x3(1024, 1024),
            _conv3x3(1024, 1024),
        ]

        return nn.Sequential(*layers)
    
    def forward(self, x):

        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.conn(x)
        
        x = x.view(x.shape[0], -1, self.S, self.S)

        return x

if __name__ == "__main__":
    
    m = YOLOv1()
    X = torch.rand(2, 3, 448, 448)
    y = m(X)
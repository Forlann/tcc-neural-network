import torch
import torch.nn as nn

# --- IMPORTAÇÃO ADICIONADA ---

from collections import OrderedDict


class CustomBackbone(nn.Module):

    def __init__(self):

        super(CustomBackbone, self).__init__()

        # (A definição da CNN continua a mesma)

        self.features = nn.Sequential(

            # Input: (B, 3, 416, 416)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 416 -> 208

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 208 -> 104

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 104 -> 52

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 52 -> 26

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 26 -> 13      
            # Output: (B, 128, 13, 13)
        )
        # O Faster R-CNN precisa saber o número de canais de saída
        self.out_channels = 128

    def forward(self, x):
        # Passa os dados pela CNN
        x = self.features(x)
        return OrderedDict([('0', x)])

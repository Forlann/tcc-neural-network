import torch
import torch.nn as nn
from collections import OrderedDict

class TccBackbone(nn.Module):
    def __init__(self, layers_config=[1, 1, 1, 1, 1]):
        super(TccBackbone, self).__init__()
        
        # channels = [3, 32, 64, 128, 256, 256] 
        channels = [3, 16, 32, 64, 128, 128] 

        self.features = nn.Sequential()

        # Loop para criar os 5 blocos da pirâmide
        for i in range(5):
            in_c = channels[i]
            out_c = channels[i+1]
            
            num_layers = layers_config[i]

            for j in range(num_layers):
                # Lógica: Apenas a primeira camada do bloco faz a transição de canais (ex: 32 -> 64).
                # Todas as outras camadas extras (j > 0) mantêm a largura fixa (ex: 64 -> 64).
                # Isso aumenta a profundidade sem alterar a largura final do bloco.
                
                current_in = in_c if j == 0 else out_c
                
                # Convolução
                self.features.add_module(
                    f'block{i+1}_conv{j+1}', 
                    nn.Conv2d(current_in, out_c, kernel_size=3, padding=1, bias=False)
                )
                
                # Batch Norm (Estabilidade)
                self.features.add_module(
                    f'block{i+1}_bn{j+1}', 
                    nn.BatchNorm2d(out_c)
                )
                
                # ReLU
                self.features.add_module(
                    f'block{i+1}_relu{j+1}', 
                    nn.ReLU(inplace=True)
                )
            
            # MaxPool
            self.features.add_module(f'block{i+1}_pool', nn.MaxPool2d(2, 2))

        self.out_channels = 128

    def forward(self, x):
        x = self.features(x)
        return OrderedDict([('0', x)])
import torch
from torchvision.datasets import CocoDetection
import numpy as np

class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        """
        root: Caminho para a pasta das imagens
        annFile: Caminho para o arquivo .json das anotações
        transform: As transformações (Resize, ToTensor) que já usávamos
        """
        super().__init__(root, annFile)
        self._transforms = transform

    def __getitem__(self, idx):
        # 1. Carrega imagem e anotação bruta do COCO
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        # 2. Se a imagem não tiver anotações (vazio), retorna alvo vazio
        # Isso evita erros se tiver fotos "inúteis" no dataset
        if len(target) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            # 3. Extrai as coordenadas e converte de COCO [x, y, w, h] 
            # para PyTorch [x1, y1, x2, y2]
            boxes = []
            labels = []
            area = []
            iscrowd = []

            for obj in target:
                xmin = obj['bbox'][0]
                ymin = obj['bbox'][1]
                w = obj['bbox'][2]
                h = obj['bbox'][3]
                
                xmax = xmin + w
                ymax = ymin + h
                
                boxes.append([xmin, ymin, xmax, ymax])
                # Importante: O COCO geralmente começa categorias em 1.
                # O PyTorch Faster R-CNN espera labels int64.
                labels.append(obj['category_id']) 
                area.append(obj['area'])
                iscrowd.append(obj['iscrowd'])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # 4. Monta o dicionário final que o Faster R-CNN exige
        final_target = {}
        final_target["boxes"] = boxes
        final_target["labels"] = labels
        final_target["image_id"] = torch.tensor([image_id])
        final_target["area"] = area
        final_target["iscrowd"] = iscrowd

        # 5. Aplica as transformações (Resize, ToTensor)
        if self._transforms is not None:
            # Nota: Em projetos avançados, transforms também devem ajustar as Bboxes (se houver resize).
            # Como seu transform atual é simples, aplicamos na imagem. 
            # Mas CUIDADO: Se o resize mudar a proporção da imagem, as caixas vão ficar tortas.
            # O ideal é que seu transform saiba lidar com targets, mas vamos manter simples por enquanto.
            img = self._transforms(img)

        return img, final_target
# =============================================================
# dataset_coco.py
# =============================================================
# Carrega imagens e anotações no formato COCO (.json)
# para uso com modelos de detecção de objetos no PyTorch.
# =============================================================

import os
import json
import torch
from PIL import Image
from torchvision import transforms


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        """
        Parâmetros:
        -----------
        root : str
            Caminho da pasta que contém as imagens (ex: data/merged/train/images).
        annotation_file : str
            Caminho do arquivo COCO JSON com as anotações.
        transforms : callable, opcional
            Funções de transformação aplicadas às imagens.
        """
        self.root = root
        self.transforms = transforms

        with open(annotation_file, "r", encoding="utf-8") as f:
            self.coco = json.load(f)

        # Dicionário auxiliar: id → info da imagem
        self.images = {img["id"]: img for img in self.coco["images"]}

        # Índice rápido: image_id → anotações
        self.annotations = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Lista de todos os image_ids
        self.image_ids = list(self.images.keys())

    def __getitem__(self, idx):
        """
        Retorna imagem e anotações no formato compatível com PyTorch.
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        anns = self.annotations.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])  # IDs já definidos globalmente

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_ids)


# =============================================================
# Função auxiliar: get_transform
# =============================================================
def get_transform(train=True):
    """
    Define transformações de imagem:
    - Sempre converte para tensor
    - Aplica flip horizontal aleatório durante o treino
    """
    transforms_list = [transforms.ToTensor()]
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms_list)

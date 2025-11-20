# =============================================================
# train.py
# =============================================================
# Treinamento do modelo SSD (VGG16 backbone) usando dataset COCO.
# =============================================================

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.ssd_vgg16 import create_ssd_model
from dataset import COCODataset, get_transform  # usa o dataset COCO agora

# =============================================================
# CONFIGURAÇÕES GERAIS
# =============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7  # 6 classes + 1 background
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-4

DATA_ROOT = "data/merged"
SAVE_PATH = "ssd_vgg16_merged.pth"

print("🚀 Usando dispositivo:", DEVICE)

# =============================================================
# PREPARAÇÃO DO DATASET
# =============================================================

train_dataset = COCODataset(
    root=os.path.join(DATA_ROOT, "train", "images"),
    annotation_file=os.path.join(DATA_ROOT, "train", "_annotations_merged.coco.json"),
    transforms=get_transform(train=True)
)

val_dataset = COCODataset(
    root=os.path.join(DATA_ROOT, "valid", "images"),
    annotation_file=os.path.join(DATA_ROOT, "valid", "_annotations_merged.coco.json"),
    transforms=get_transform(train=False)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

print(f"📊 Imagens de treino: {len(train_dataset)}, validação: {len(val_dataset)}")

# =============================================================
# DEFINIÇÃO DO MODELO SSD
# =============================================================
model = create_ssd_model(num_classes=NUM_CLASSES).to(DEVICE)

# =============================================================
# OTIMIZADOR
# =============================================================
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

# =============================================================
# LOOP DE TREINAMENTO
# =============================================================
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    with tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS}", unit="batch") as tepoch:
        for images, targets in tepoch:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()
            epoch_loss += loss_value
            tepoch.set_postfix(loss=loss_value)

    avg_loss = epoch_loss / len(train_loader)
    print(f"→ Época [{epoch+1}/{EPOCHS}] concluída | Loss médio: {avg_loss:.4f}")

# =============================================================
# SALVAMENTO DO MODELO
# =============================================================
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ Treinamento finalizado. Modelo salvo em: {SAVE_PATH}")

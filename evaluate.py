# =============================================================
# evaluate.py (versão compatível com métricas YOLO e dataset COCO)
# =============================================================

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from models.ssd_vgg16 import create_ssd_model
from dataset import COCODataset, get_transform  # ✅ usa dataset_coco.py

# =============================================================
# CONFIGURAÇÕES
# =============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8  # 7 classes + background
BATCH_SIZE = 4

DATA_ROOT = "data/merged"
MODEL_PATH = "ssd_vgg16_merged.pth"

# =============================================================
# FUNÇÕES AUXILIARES
# =============================================================
def compute_iou(box1, box2):
    """Calcula IoU entre duas caixas [xmin, ymin, xmax, ymax]."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def mean_average_precision(preds, gts, iou_thresholds):
    """Calcula mAP médio para múltiplos limiares de IoU."""
    aps = []
    for thr in iou_thresholds:
        tp, fp, fn = 0, 0, 0
        for p, g in zip(preds, gts):
            matched = set()
            for pb in p:
                found = False
                for i, gb in enumerate(g):
                    if i in matched:
                        continue
                    if compute_iou(pb, gb) >= thr:
                        tp += 1
                        matched.add(i)
                        found = True
                        break
                if not found:
                    fp += 1
            fn += len(g) - len(matched)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        aps.append((precision, recall))
    precisions = [p for p, _ in aps]
    recalls = [r for _, r in aps]
    map50 = precisions[0]  # IoU=0.5
    map5095 = np.mean(precisions)  # IoU 0.5→0.95
    return map50, map5095, np.mean(precisions), np.mean(recalls)

# =============================================================
# CARREGAR MODELO E DADOS
# =============================================================
test_dataset = COCODataset(
    root=os.path.join(DATA_ROOT, "test", "images"),  # ✅ pasta com imagens
    annotation_file=os.path.join(DATA_ROOT, "test", "_annotations_merged.coco.json"),  # ✅ JSON COCO
    transforms=get_transform(train=False)
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

model = create_ssd_model(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"✅ Modelo SSD carregado e pronto para avaliação ({len(test_dataset)} imagens)\n")

# =============================================================
# LOOP DE AVALIAÇÃO
# =============================================================
pred_boxes = []
gt_boxes = []

with torch.no_grad():
    for imgs, targets in tqdm(test_loader, desc="Avaliando SSD", unit="batch"):
        imgs = [img.to(DEVICE) for img in imgs]
        outputs = model(imgs)
        for output, target in zip(outputs, targets):
            preds = output["boxes"].cpu().numpy()
            gts = target["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            preds = preds[scores > 0.3]  # ✅ pode ajustar o threshold aqui
            pred_boxes.append(preds)
            gt_boxes.append(gts)

# =============================================================
# CÁLCULO DAS MÉTRICAS
# =============================================================
iou_thresholds = np.arange(0.5, 1.0, 0.05)
map50, map5095, precision, recall = mean_average_precision(pred_boxes, gt_boxes, iou_thresholds)

# =============================================================
# RESULTADOS
# =============================================================
print("\n===== MÉTRICAS SSD (compatíveis com YOLO) =====")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"mAP@0.5:     {map50:.4f}")
print(f"mAP@0.5–0.95:{map5095:.4f}")
print("================================================")

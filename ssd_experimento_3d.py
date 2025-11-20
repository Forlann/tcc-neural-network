# ssd_experimento_3d.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataset import COCODataset, get_transform
from models.ssd_vgg16 import create_ssd_model

# -----------------------
# CONFIGURAÇÕES (ajuste se necessário)
# -----------------------
DATA_ROOT = "data/merged"                 # raiz com train/ valid/ test
TRAIN_ROOT = os.path.join(DATA_ROOT, "train", "images")
TRAIN_JSON = os.path.join(DATA_ROOT, "train", "_annotations_merged.coco.json")

VAL_ROOT = os.path.join(DATA_ROOT, "valid", "images")
VAL_JSON = os.path.join(DATA_ROOT, "valid", "_annotations_merged.coco.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7          # 6 classes reais + background
BATCH_SIZE = 24
LEARNING_RATE = 1e-4

# experimento (padrões, ajuste conforme HW)
TOTAL_EPOCHS = 150
# checkpoints em que vamos calcular métricas (10,20,...,TOTAL_EPOCHS)
CHECKPOINTS = list(range(10, TOTAL_EPOCHS + 1, 10))

# porcentagens do dataset a testar (de 5% a 100% em 20 passos)
PORCENTAGENS = np.linspace(0.05, 1.0, 20)

# CSV de saída
CSV_OUT = "ssd_3d_metrics.csv"

# score threshold para filtrar predições durante validação
SCORE_TH = 0.5


# -----------------------
# FUNÇÕES DE MÉTRICAS (compatível com evaluate.py)
# -----------------------
def compute_iou(box1, box2):
    """
    IoU entre duas caixas [xmin, ymin, xmax, ymax]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def mean_average_precision(preds, gts, iou_thresholds):
    """
    Versão simplificada do cálculo de mAP usada no projeto:
    - preds: list of arrays (N_pred_boxes x 4) por imagem
    - gts:   list of arrays (N_gt_boxes x 4) por imagem
    - iou_thresholds: iterable de limiares (ex: np.arange(0.5, 1.0, 0.05))
    Retorna: map50, map50_95, mean_precision, mean_recall
    """
    aps = []
    for thr in iou_thresholds:
        tp, fp, fn = 0, 0, 0
        for p_boxes, g_boxes in zip(preds, gts):
            matched = set()
            # para cada predicted box, procura GT correspondente
            for pb in p_boxes:
                found = False
                for i, gb in enumerate(g_boxes):
                    if i in matched:
                        continue
                    if compute_iou(pb, gb) >= thr:
                        tp += 1
                        matched.add(i)
                        found = True
                        break
                if not found:
                    fp += 1
            fn += max(0, len(g_boxes) - len(matched))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        aps.append((precision, recall))
    precisions = [p for p, _ in aps]
    recalls = [r for _, r in aps]
    map50 = precisions[0] if len(precisions) > 0 else 0.0
    map50_95 = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    mean_precision = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    mean_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    return map50, map50_95, mean_precision, mean_recall


# -----------------------
# VALIDAÇÃO (usa model e val_loader)
# -----------------------
def validar_modelo(model, val_loader, device, score_th=SCORE_TH):
    model.eval()
    pred_boxes = []
    gt_boxes = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            for out, tgt in zip(outputs, targets):
                # out: dict com 'boxes','scores','labels'
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                # filtra por score
                boxes = boxes[scores > score_th]
                pred_boxes.append(boxes)
                # gt
                gts = tgt["boxes"].cpu().numpy()
                gt_boxes.append(gts)

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    map50, map50_95, precision, recall = mean_average_precision(pred_boxes, gt_boxes, iou_thresholds)
    return precision, recall, map50, map50_95


# -----------------------
# FUNÇÃO PRINCIPAL DO EXPERIMENTO 3D
# -----------------------
def rodar_experimento_ssd():
    print("🔬 Iniciando experimento SSD 3D")
    # dataset completo (train) e validação
    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    train_dataset_full = COCODataset(root=TRAIN_ROOT, annotation_file=TRAIN_JSON, transforms=train_transform)
    val_dataset = COCODataset(root=VAL_ROOT, annotation_file=VAL_JSON, transforms=val_transform)

    total_imgs = len(train_dataset_full)
    print(f"Dataset de treino completo: {total_imgs} imagens")
    print(f"Dataset de validação: {len(val_dataset)} imagens")

    # monta lista de tamanhos a testar
    tamanhos = [max(1, int(total_imgs * p)) for p in PORCENTAGENS]

    # dataloader de validação fixo
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    resultados = []

    for qtd_imgs in tamanhos:
        if qtd_imgs > total_imgs:
            qtd_imgs = total_imgs

        print("\n" + "=" * 60)
        print(f"🔁 Cenário: usar {qtd_imgs} imagens para treino (de {total_imgs})")
        print("=" * 60)

        # criar subset aleatório fixo para este cenário
        indices = torch.randperm(total_imgs)[:qtd_imgs]
        subset_train = Subset(train_dataset_full, indices)
        train_loader = DataLoader(subset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

        # reinicia modelo (treina do zero)
        model = create_ssd_model(num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

        # Treinamento contínuo até TOTAL_EPOCHS
        for epoch in range(1, TOTAL_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            batches = 0

            with tqdm(train_loader, desc=f"[Imgs {qtd_imgs}] Época {epoch}/{TOTAL_EPOCHS}", leave=False) as pbar:
                for imgs, targets in pbar:
                    imgs = [img.to(DEVICE) for img in imgs]
                    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                    loss_dict = model(imgs, targets)
                    loss = sum(v for v in loss_dict.values())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batches += 1
                    pbar.set_postfix_str(f"loss={loss.item():.4f}")

            media_loss = epoch_loss / (batches if batches > 0 else 1)
            print(f"[Imgs {qtd_imgs}] Época {epoch} finalizada | Loss médio: {media_loss:.4f}")

            # checkpoints: rodar validação e salvar métricas
            if epoch in CHECKPOINTS:
                print(f"[Imgs {qtd_imgs}] → Checkpoint (época {epoch}) - Validando...")
                precision, recall, map50, map50_95 = validar_modelo(model, val_loader, DEVICE, score_th=SCORE_TH)
                print(f"    precision={precision:.4f} recall={recall:.4f} map50={map50:.4f} map50_95={map50_95:.4f}")

                resultados.append({
                    "modelo": "SSD_vgg16",
                    "num_imagens": int(qtd_imgs),
                    "epoca": int(epoch),
                    "loss": float(media_loss),
                    "precision": float(precision),
                    "recall": float(recall),
                    "map50": float(map50),
                    "map50_95": float(map50_95)
                })

                # salva parcial em CSV
                pd.DataFrame(resultados).to_csv(CSV_OUT, index=False)
                print(f"    ✅ Parciais salvas em: {CSV_OUT}")

    # salva final
    pd.DataFrame(resultados).to_csv(CSV_OUT, index=False)
    print(f"\n✅ Experimento concluído. CSV final salvo em: {CSV_OUT}")


if __name__ == "__main__":
    rodar_experimento_ssd()

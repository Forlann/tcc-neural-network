import time
import os
import glob
import json
from collections import Counter
from ultralytics import YOLO
import torch
import numpy as np


# ========== 1️⃣ Converter segmentos em boxes (YOLO detect) ==========
def fix_segments_to_boxes(labels_dir):
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    converted = 0
    for file in txt_files:
        new_lines = []
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # linha inválida
                cls = parts[0]
                coords = list(map(float, parts[1:5]))  # mantém apenas x_center, y_center, w, h
                new_lines.append(f"{cls} {' '.join(map(str, coords))}\n")
        with open(file, "w") as f:
            f.writelines(new_lines)
        converted += 1
    print(f"[✓] Convertidos/removidos segmentos em {converted} arquivos de {labels_dir}")


# ========== 2️⃣ Remover labels duplicados e arquivos vazios ==========
def clean_labels(labels_dir):
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    removed_files = 0
    fixed_files = 0

    for file in txt_files:
        with open(file, "r") as f:
            lines = list(set([l.strip() for l in f if l.strip()]))

        if not lines:
            os.remove(file)
            removed_files += 1
            continue

        with open(file, "w") as f:
            f.write("\n".join(lines) + "\n")
        fixed_files += 1

    print(f"[✓] Limpou {fixed_files} arquivos e removeu {removed_files} vazios em {labels_dir}")


# ========== 3️⃣ Contar instâncias por classe ==========
def count_labels(labels_dir, class_names):
    counter = Counter()
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    for file in txt_files:
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    counter[cls] += 1
    print("\n📊 Contagem de instâncias por classe:")
    for i, name in class_names.items():
        print(f"  {name}: {counter[i]}")
    print()


# ========== 4️⃣ Treinamento e Métricas ==========
def main():
    print("🚀 Iniciando script YOLOv11 com auditoria de labels...\n")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}\n")

    model = YOLO("yolo11n.pt")
    data_yaml = "datasets.yaml"
    save_metrics_path = "metrics_summary.json"

    # ===== ETAPA DE LIMPEZA DE LABELS =====
    print("🧹 Corrigindo e auditando labels...")
    fix_segments_to_boxes("datasets/train/labels")
    fix_segments_to_boxes("datasets/valid/labels")
    clean_labels("datasets/train/labels")
    clean_labels("datasets/valid/labels")
    count_labels("datasets/train/labels", model.names)

    # ===== TREINAMENTO =====
    print("🏋️ Iniciando treinamento...\n")
    start_train = time.time()

    model.train(
        data=data_yaml,
        epochs=1,
        imgsz=640,
        project="runs/train",
        name="yolo_multiclass",
        batch=30,
        plots=True,
        patience=20,
        workers=4
    )

    end_train = time.time()
    training_time = end_train - start_train
    print(f"\n[✓] Tempo total de treinamento: {training_time:.2f} segundos")

    # ===== VALIDAÇÃO =====
    print("\n🔎 Validando modelo...")
    results = model.val()
    metrics_dict = results.results_dict

    # ===== MÉTRICAS POR CLASSE =====
    class_metrics = []
    names = model.names

    for i, name in names.items():
        class_metrics.append({
            "classe": name,
            "precision": float(results.box.precision[i]),
            "recall": float(results.box.recall[i]),
            "mAP50": float(results.box.map50[i]),
            "mAP50-95": float(results.box.map[i]),
        })

    # ===== SALVAR MÉTRICAS =====
    summary = {
        "tempo_de_treinamento_s": round(training_time, 2),
        "metricas_por_classe": class_metrics,
        "metricas_gerais": {
            "precision": float(metrics_dict['metrics/precision(B)']),
            "recall": float(metrics_dict['metrics/recall(B)']),
            "mAP50": float(metrics_dict['metrics/mAP50(B)']),
            "mAP50-95": float(metrics_dict['metrics/mAP50-95(B)'])
        }
    }

    with open(save_metrics_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[✓] Métricas salvas em: {save_metrics_path}")
    print("✅ Processo concluído com sucesso!")


if __name__ == "__main__":
    main()

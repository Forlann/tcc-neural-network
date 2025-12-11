import time
from ultralytics import YOLO
import torch
import numpy as np
import os
import json




def main():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    model = YOLO("yolo11n.pt")
    data_yaml = "datasets.yaml"
    save_metrics_path = "metrics_summary.json"

    # ========= Etapa 1: Treinamento =========
    start_train = time.time()

    model.train(
        data=data_yaml,
        epochs=1,
        imgsz=640,
        project="runs/train",
        name="yolo_multiclass",
        batch = 0.75,
        plots = True,
        patience = 20,
        workers=4
    )

    end_train = time.time()
    training_time = end_train - start_train
    print(f"[✓] Tempo total de treinamento: {training_time:.2f} segundos")

    # ========= Etapa 2: Validação =========
    results = model.val()
    metrics_dict = results.results_dict

    # ========= Etapa 3: Métricas por Classe =========
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

    print(f"[✓] Métricas salvas em: {save_metrics_path}")

if __name__ == "__main__":
    main()

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import time
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from CocoDataset import CustomCocoDataset
from functions import detection_collate_fn
from TCCBackbone import TccBackbone 
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# --- CONFIGURAÇÕES GERAIS ---
TRAIN_IMG_DIR = "C:/Users/leonardo/Desktop/Neural Tcc kaue/data/merged/train/images"
TRAIN_JSON = "C:/Users/leonardo/Desktop/Neural Tcc kaue/data/merged/train/_annotations_merged.coco.json"

VAL_IMG_DIR = "C:/Users/leonardo/Desktop/Neural Tcc kaue/data/merged/valid/images"
VAL_JSON = "C:/Users/leonardo/Desktop/Neural Tcc kaue/data/merged/valid/_annotations_merged.coco.json"

NUM_CLASSES = 6
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_EPOCAS = 100

# --- CONFIGURAÇÕES DOS MODELOS ---
MODEL_CONFIGS = [
    #{
   #     "name": "7_Layers",
   #     "config": [2, 2, 1, 1, 1] 
   # },
    {
        "name": "8_Layers",
        "config": [2, 2, 2, 1, 1]
    }
    # {
    #     "name": "10_Layers",
    #     "config": [2, 2, 2, 2, 2]
    # },
    # {
    #     "name": "15_Layers",
    #     "config": [3, 3, 3, 3, 3]
    # }
]

# --- FUNÇÃO DE MODELO ---
def get_detection_model(num_classes, layers_config):
    """
    Agora recebe layers_config para criar backbones de profundidades diferentes.
    """
    # Instancia o backbone passando a configuração da lista
    backbone = TccBackbone(layers_config=layers_config)
    
    # backbone.out_channels = 256 
    backbone.out_channels = 128 
    
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    
    model = FasterRCNN(backbone, num_classes=91, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    
    # Ajusta o cabeçalho (Head) para o número de classes do seu dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    return model

# --- FUNÇÃO DE VALIDAÇÃO ---
def validar_e_calcular_map(model, val_loader, device):
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)
            metric.update(preds, targets)
    result = metric.compute()
    return result['map'].item()

def rodar_experimento():
    print(f"--- INICIANDO EXPERIMENTO DE LAYERS ---")
    print(f"Device: {DEVICE}")
    print(f"Total Épocas por modelo: {TOTAL_EPOCAS}")

    # 1. Setup de Dados (Carrega uma única vez)
    transform_padrao = transforms.Compose([
        transforms.Resize((640, 640)), 
        transforms.ToTensor()
    ])
    
    # Usamos o dataset COMPLETO para comparação justa
    train_dataset = CustomCocoDataset(root=TRAIN_IMG_DIR, annFile=TRAIN_JSON, transform=transform_padrao)
    val_dataset = CustomCocoDataset(root=VAL_IMG_DIR, annFile=VAL_JSON, transform=transform_padrao)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=detection_collate_fn)
    
    print(f"Imagens de Treino: {len(train_dataset)}")
    print(f"Imagens de Validação: {len(val_dataset)}")
    
    # Checkpoints para salvar métricas (a cada 10 épocas)
    checkpoints_epocas = list(range(10, TOTAL_EPOCAS + 1, 10))
    resultados = [] 

    # --- LOOP EXTERNO: VARIA A ARQUITETURA (LAYERS) ---
    for scenario in MODEL_CONFIGS:
        model_name = scenario['name']
        layer_cfg = scenario['config']
        
        print(f"\n==================================================")
        print(f"TREINANDO MODELO: {model_name}")
        print(f"Configuração: {layer_cfg}")
        print(f"==================================================")
        
        # Instancia o modelo com a configuração específica
        model = get_detection_model(NUM_CLASSES, layers_config=layer_cfg)
        model.to(DEVICE)
        
        # Otimizador (Reinicia para cada modelo para ser justo)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        start_time = time.time() # Para medir tempo total do modelo
        
        # --- LOOP INTERNO: ÉPOCAS ---
        for epoch in range(1, TOTAL_EPOCAS + 1):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            batches_rodados = 0 

            for images, targets in train_loader:
                images = list(img.to(DEVICE) for img in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                loss = sum(l for l in loss_dict.values())
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batches_rodados += 1
            
            epoch_end = time.time()
            tempo_epoca = epoch_end - epoch_start
            
            # Cálculo da média
            if batches_rodados > 0:
                media_loss = running_loss / batches_rodados
            else:
                media_loss = 0.0

            print(f"[{model_name}] Época {epoch}/{TOTAL_EPOCAS} | Loss: {media_loss:.4f} | Tempo: {tempo_epoca:.1f}s")
            
            # --- VALIDAÇÃO E COLETA DE DADOS ---
            if epoch in checkpoints_epocas:
                print(f"   > Validando...", end="")
                map_score = validar_e_calcular_map(model, val_loader, DEVICE)
                print(f" mAP: {map_score:.4f}")
                
                resultados.append({
                    'modelo': model_name,
                    'config_layers': str(layer_cfg), # Salva a config como string
                    'epoca': epoch,
                    'loss': media_loss,
                    'mAP': map_score,
                    'tempo_acumulado_min': (time.time() - start_time) / 60
                })
                
                # Salva parcial a cada validação para não perder dados se acabar a luz
                pd.DataFrame(resultados).to_csv('resultados_parciais_layers.csv', index=False)

    # --- FIM DO EXPERIMENTO ---
    print("\nExperimento finalizado!")
    df = pd.DataFrame(resultados)
    df.to_csv('resultados_finais_layers_comparison.csv', index=False)
    print("Arquivo 'resultados_finais_layers_comparison.csv' gerado.")

if __name__ == '__main__':
    rodar_experimento()
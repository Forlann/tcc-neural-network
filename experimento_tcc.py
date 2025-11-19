import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import pandas as pd
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
# from YoloDataset import YoloDataset
from CocoDataset import CustomCocoDataset
from functions import detection_collate_fn
from CustomBackbone import CustomBackbone 
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# --- Configurações ---
TRAIN_IMG_DIR = "D:/dev/faculdade/TCC/REDE_NEURAL/v0.2/dataset_merged/merged/merged/train/images"
TRAIN_JSON = "D:/dev/faculdade/TCC/REDE_NEURAL/v0.2/dataset_merged/merged/merged/train/_annotations_merged.coco.json"

VAL_IMG_DIR = "D:/dev/faculdade/TCC/REDE_NEURAL/v0.2/dataset_merged/merged/merged/valid/images"
VAL_JSON = "D:/dev/faculdade/TCC/REDE_NEURAL/v0.2/dataset_merged/merged/merged/valid/_annotations_merged.coco.json"

NUM_CLASSES = 6
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Função de Modelo ---
def get_detection_model(num_classes):
    backbone = CustomBackbone()
    anchor_generator = AnchorGenerator(sizes=((128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=91, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model

# --- Função de Validação ---
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

# =============================================================================
# O NOVO GERENTE DO EXPERIMENTO (Lógica Otimizada)
# =============================================================================

def rodar_experimento():
    # 1. Setup Inicial
    # Se suas imagens originais não são 640x640, as caixas vão ficar erradas
    # porque redimensionamos a foto mas não recalculamos as coordenadas.
    # Para TCC, o ideal é usar transforms que ajustem as caixas, mas para testar agora:
    transform_padrao = transforms.Compose([
        transforms.Resize((640, 640)), 
        transforms.ToTensor()
    ])
    
    # Instanciando o Dataset COCO
    full_train_dataset = CustomCocoDataset(
        root=TRAIN_IMG_DIR, 
        annFile=TRAIN_JSON, 
        transform=transform_padrao
    )
    
    val_dataset = CustomCocoDataset(
        root=VAL_IMG_DIR, 
        annFile=VAL_JSON, 
        transform=transform_padrao
    )
    # -----------------------------------

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=detection_collate_fn)
    
    # 2. DEFINIÇÃO DOS EIXOS DO GRÁFICO
    # Eixo Z: Quantidade de Imagens (ex: 100, 500, 1000...)
    # Se seu dataset for menor que 1000, ajuste esses números!
    qts_dataset = len(full_train_dataset)
    
    qtd_imagens_lista = [
                            int(qts_dataset*0.05),       # 5%
                            int(qts_dataset*0.10),       # 10%
                            int(qts_dataset*0.15),       # 15% 
                            int(qts_dataset*0.20),       # 20% 
                            int(qts_dataset*0.25),       # 25% 
                            int(qts_dataset*0.30),       # 30% 
                            int(qts_dataset*0.35),       # 35% 
                            int(qts_dataset*0.40),       # 40% 
                            int(qts_dataset*0.45),       # 45% 
                            int(qts_dataset*0.50),       # 50% 
                            int(qts_dataset*0.55),       # 55% 
                            int(qts_dataset*0.60),       # 60% 
                            int(qts_dataset*0.65),       # 65% 
                            int(qts_dataset*0.70),       # 70% 
                            int(qts_dataset*0.75),       # 75% 
                            int(qts_dataset*0.80),       # 80% 
                            int(qts_dataset*0.85),       # 85% 
                            int(qts_dataset*0.90),       # 90% 
                            int(qts_dataset*0.95),       # 95% 
                            int(len(full_train_dataset)) # 100%
                        ]
    
    # Eixo X: Épocas Totais
    TOTAL_EPOCAS = 150
    
    # Checkpoints: Em quais épocas queremos salvar o dado? (10, 20, 30... 100)
    # O range(10, 101, 10) gera: [10, 20, 30, ..., 100]
    checkpoints_epocas = list(range(10, TOTAL_EPOCAS + 1, 10))

    resultados = [] 

    # --- LOOP EXTERNO: VARIA O TAMANHO DO DATASET ---
    for qtd_imgs in qtd_imagens_lista:
        if qtd_imgs > len(full_train_dataset):
            qtd_imgs = len(full_train_dataset) # Ajuste de segurança

        print(f"INICIANDO CENÁRIO: Dataset com {qtd_imgs} Imagens <<<")
        
        # Cria o subset fixo para este cenário
        indices = torch.randperm(len(full_train_dataset))[:qtd_imgs].tolist()
        subset_train = Subset(full_train_dataset, indices)
        train_loader = DataLoader(subset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=detection_collate_fn)
        
        # REINICIA O MODELO (Aqui sim, resetamos porque mudou o dataset)
        model = get_detection_model(NUM_CLASSES)
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # --- LOOP INTERNO: TREINAMENTO CONTÍNUO ---
        for epoch in range(1, TOTAL_EPOCAS + 1):
            # 1. Treina uma época
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
            
            # CÁLCULO DA MÉDIA
            if batches_rodados > 0:
                media_loss = running_loss / batches_rodados
            else:
                media_loss = 0.0
                print("⚠️ AVISO: O DataLoader não retornou nenhuma imagem nesta época!")

            print(f"Época {epoch} | Loss Médio: {media_loss:.4f}")
            
            # 2. Verifica se é hora de coletar dados para o gráfico
            if epoch in checkpoints_epocas:
                print(f" [Dataset {qtd_imgs} imgs | Época {epoch}] Validando...", end="")
                
                # AVALIA O MODELO ATUAL
                map_score = validar_e_calcular_map(model, val_loader, DEVICE)
                print(f" mAP Final: {map_score:.4f}")
                
                resultados.append({
                    'modelo': 'Custom_FasterRCNN',
                    'qtd_imagens': qtd_imgs,
                    'epoca': epoch,
                    'loss': media_loss,  # Adicionei o loss no CSV também
                    'mAP': map_score
                })
                
                pd.DataFrame(resultados).to_csv('resultados_parciais_tcc.csv', index=False)

    # Salva final
    df = pd.DataFrame(resultados)
    df.to_csv('resultados_finais_tcc_3d.csv', index=False)
    print("/n✅ Dados gerados com sucesso!")

if __name__ == '__main__':
    rodar_experimento()
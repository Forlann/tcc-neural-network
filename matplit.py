import os
import pandas as pd
from ultralytics import YOLO
import torch

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

# Caminho do seu arquivo de dados (o unificado)
DATA_YAML = "datasets.yaml" 

# Configurações do Experimento
MODELO_BASE = "yolo11n.pt" 
TOTAL_EPOCAS = 150         
BATCH_SIZE = 0.75          
IMGSZ = 640
WORKERS = 4

# Nome do projeto para salvar logs organizados
PROJECT_NAME = "runs/experimento_tcc_3d"

# Checkpoints: Quais épocas queremos no gráfico? (Eixo X)
CHECKPOINTS_EPOCAS = list(range(10, TOTAL_EPOCAS + 1, 10))

# Frações do Dataset: Quanto de dado usar? (Eixo Y)
FRACOES_DATASET = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# =============================================================================
# 2. EXECUÇÃO DO EXPERIMENTO
# =============================================================================

def rodar_experimento_yolo():
    print(f"🚀 Iniciando experimento 3D para {len(FRACOES_DATASET)} cenários de dados...")
    
    resultados_finais = []

    # --- LOOP EXTERNO: VARIA O TAMANHO DO DATASET (Eixo Y) ---
    for i, fraction in enumerate(FRACOES_DATASET):
        
        nome_execucao = f"treino_frac_{int(fraction*100)}pct"
        print(f"\n\n>>> [CENÁRIO {i+1}/{len(FRACOES_DATASET)}] Treinando com {int(fraction*100)}% dos dados...")

        model = YOLO(MODELO_BASE)

        # 2. TREINA O MODELO (Calcula mAP a cada época automaticamente)
        results = model.train(
            data=DATA_YAML,
            epochs=TOTAL_EPOCAS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=nome_execucao,
            workers=WORKERS,
            fraction=fraction,
            exist_ok=True,
            plots=False,
            verbose=False 
        )

        # 3. EXTRAÇÃO DE DADOS (Pós-Treino)
        csv_path = os.path.join(PROJECT_NAME, nome_execucao, "results.csv")
        
        if os.path.exists(csv_path):
            df_yolo = pd.read_csv(csv_path)
            # Limpa espaços em branco nos nomes das colunas
            df_yolo.columns = [c.strip() for c in df_yolo.columns]

            print(f"   -> Extraindo mAP das épocas: {CHECKPOINTS_EPOCAS}")

            for epoca_alvo in CHECKPOINTS_EPOCAS:
                # Tenta encontrar a linha da época exata
                linha = df_yolo[df_yolo['epoch'] == epoca_alvo]
                
                # Se não achar (às vezes o CSV é indexado em 0, ex: 0 a 99), tenta epoca - 1
                if linha.empty:
                     linha = df_yolo[df_yolo['epoch'] == (epoca_alvo - 1)]

                if not linha.empty:
                    # Captura as métricas desejadas
                    map50 = linha['metrics/mAP50(B)'].values[0] 
                    map50_95 = linha['metrics/mAP50-95(B)'].values[0] # <--- O "MAP" mais completo
                    precision = linha['metrics/precision(B)'].values[0]
                    
                    print(f"      [Epoch {epoca_alvo}] mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}")

                    resultados_finais.append({
                        'dataset_fracao': fraction,
                        'dataset_porcentagem': int(fraction*100),
                        'epoca': epoca_alvo,
                        'precision': precision,
                        'mAP50': map50,
                        'mAP50-95': map50_95 # Adicionado para garantir
                    })
        else:
            print(f"⚠️ ERRO: Não encontrei o arquivo {csv_path}")

        pd.DataFrame(resultados_finais).to_csv('resultados_3d_yolo_parcial.csv', index=False)

    print("\n✅ Experimento Finalizado!")
    df_final = pd.DataFrame(resultados_finais)
    df_final.to_csv('resultados_finais_tcc_3d.csv', index=False)
    print(f"Arquivo salvo: resultados_finais_tcc_3d.csv")

if __name__ == "__main__":
    rodar_experimento_yolo()
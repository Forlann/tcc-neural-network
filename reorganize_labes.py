import os

def remap_class_ids(diretorio_labels, offset):
    """
    Percorre um diretório de labels YOLO (.txt) e adiciona um 'offset'
    a todos os IDs de classe.

    Args:
        diretorio_labels (str): O caminho para a pasta com os .txt
        offset (int): O valor a ser somado a cada ID de classe
    """
    print(f"--- Iniciando remapeamento em: {diretorio_labels} [Offset: +{offset}] ---")

    # Offset 0 significa que não há mudança, pulamos o processamento
    if offset == 0:
        print("Offset é 0, nenhum remapeamento necessário. Pulando.")
        print("---------------------------------------------------\n")
        return

    # Verifica se o diretório existe
    if not os.path.isdir(diretorio_labels):
        print(f"ERRO: O diretório não existe. Pulando: {diretorio_labels}")
        print("---------------------------------------------------\n")
        return

    arquivos_modificados = 0
    total_detecções_remapeadas = 0

    for nome_arquivo in os.listdir(diretorio_labels):
        if nome_arquivo.endswith(".txt"):
            caminho_arquivo = os.path.join(diretorio_labels, nome_arquivo)
            novas_linhas = []
            
            try:
                with open(caminho_arquivo, 'r') as f:
                    linhas = f.readlines()
                
                if not linhas:
                    continue  # Pula arquivos vazios

                for linha in linhas:
                    partes = linha.strip().split()
                    
                    if len(partes) >= 5:
                        # 1. Converte o ID da classe para inteiro
                        old_class_id = int(partes[0])
                        
                        # 2. A MÁGICA: Adiciona o offset
                        new_class_id = old_class_id + offset
                        
                        # 3. Substitui o ID antigo pelo novo (como string)
                        partes[0] = str(new_class_id)
                        
                        novas_linhas.append(" ".join(partes) + "\n")
                        total_detecções_remapeadas += 1
                
                # Reescreve o arquivo com as classes remapeadas
                with open(caminho_arquivo, 'w') as f:
                    f.writelines(novas_linhas)
                
                arquivos_modificados += 1

            except ValueError:
                print(f"Aviso: Arquivo {nome_arquivo} contém ID de classe não numérico. Pulando.")
            except Exception as e:
                print(f"Erro ao processar o arquivo {caminho_arquivo}: {e}")

    print(f"Remapeamento concluído.")
    print(f"Total de arquivos .txt modificados: {arquivos_modificados}")
    print(f"Total de detecções remapeadas: {total_detecções_remapeadas}")
    print("---------------------------------------------------\n")

# =============================================================================
# 1. CONFIGURE SEUS DATASETS E OFFSETS AQUI
# =============================================================================

# Lógica dos Offsets:
# 1. Faixa (1 classe: 0) -> offset 0 -> Classe final: 0
# 2. Veiculos (1 classe: 0) -> offset 1 -> Classe final: 1
# 3. Semaforos (3 classes: 0, 1, 2) -> offset 2 -> Classes finais: 2, 3, 4
# 4. Pessoas (1 classe: 0) -> offset 5 -> Classe final: 5

datasets_para_processar = [
    {
        "nome": "Faixa de Pedestre",
        "num_classes": 1,
        "offset": 0,  # <-- Inicia em 0
        "caminho_labels_train": "C:/Users/leonardo\Desktop/tcc/faixa_pedestre.yolov11/train/labels",
        "caminho_labels_valid": "C:/Users/leonardo\Desktop/tcc/faixa_pedestre.yolov11/valid/labels",
        "caminho_labels_test": "C:/Users/leonardo\Desktop/tcc/faixa_pedestre.yolov11/test/labels"
    },
    {
        "nome": "Veículos",
        "num_classes": 1,
        "offset": 1,  # <-- Último ID (0) + 1 = 1
        "caminho_labels_train": "C:/Users/leonardo/Desktop/tcc/veiculos.yolov11/train/labels",
        "caminho_labels_valid": "C:/Users/leonardo/Desktop/tcc/veiculos.yolov11/valid/labels",
        "caminho_labels_test": "C:/Users/leonardo/Desktop/tcc/veiculos.yolov11/test/labels"
    },
    {
        "nome": "Semáforos",
        "num_classes": 3,
        "offset": 2,  # <-- Último ID (1) + 1 = 2
        "caminho_labels_train": "C:/Users/leonardo/Desktop/tcc/semaforos.yolov11/train/labels",
        "caminho_labels_valid": "C:/Users/leonardo/Desktop/tcc/semaforos.yolov11/valid/labels",
        "caminho_labels_test": "C:/Users/leonardo/Desktop/tcc/semaforos.yolov11/test/labels"
    },
    {
        "nome": "Pessoas",
        "num_classes": 1,
        "offset": 5,  # <-- Último ID (2+2=4) + 1 = 5
        "caminho_labels_train": "C:/Users/leonardo/Desktop/tcc/pessoas.yolov11/train/labels",
        "caminho_labels_valid": "C:/Users/leonardo/Desktop/tcc/pessoas.yolov11/valid/labels",
        "caminho_labels_test": "C:/Users/leonardo/Desktop/tcc/pessoas.yolov11/test/labels"
    }
]

# String de placeholder para checagem
PLACEHOLDER = "COLOQUE_O_CAMINHO_PARA"

# =============================================================================
# 2. EXECUÇÃO DO SCRIPT
# =============================================================================

for dataset in datasets_para_processar:
    print(f"=== Processando: {dataset['nome']} ===")
    
    offset = dataset['offset']
    
    # Processa a pasta de treino
    path_train = dataset['caminho_labels_train']
    if PLACEHOLDER not in path_train:
        remap_class_ids(path_train, offset)
    else:
        print(f"AVISO: Caminho de TREINO não configurado para {dataset['nome']}. Pulando...")

    # Processa a pasta de validação
    path_valid = dataset['caminho_labels_valid']
    if PLACEHOLDER not in path_valid:
        remap_class_ids(path_valid, offset)
    else:
        print(f"AVISO: Caminho de VALIDAÇÃO não configurado para {dataset['nome']}. Pulando...")

    # Processa a pasta de teste
    path_test = dataset['caminho_labels_test']
    if PLACEHOLDER not in path_test:
        remap_class_ids(path_test, offset)
    else:
        print(f"AVISO: Caminho de TESTE não configurado para {dataset['nome']}. Pulando...")
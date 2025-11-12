import os

def unificar_classes_para_zero(diretorio_labels):
    """
    Percorre um diretório de labels YOLO (.txt) e muda todos os IDs
    de classe para '0'.

    Args:
        diretorio_labels (str): O caminho para a pasta contendo os
                                arquivos .txt (ex: .../train/labels)
    """
    print(f"--- Iniciando unificação em: {diretorio_labels} ---")
    
    arquivos_modificados = 0
    total_detecções = 0

    # Verifica se o diretório existe
    if not os.path.isdir(diretorio_labels):
        print(f"ERRO: O diretório não existe. Pulando: {diretorio_labels}")
        print("---------------------------------------------------\n")
        return

    # Lista todos os arquivos no diretório
    for nome_arquivo in os.listdir(diretorio_labels):
        # Garante que estamos processando apenas os arquivos de label
        if nome_arquivo.endswith(".txt"):
            caminho_arquivo = os.path.join(diretorio_labels, nome_arquivo)
            novas_linhas = []
            
            try:
                # 1. Lê o conteúdo original do arquivo
                with open(caminho_arquivo, 'r') as f:
                    linhas = f.readlines()
                
                if not linhas:
                    continue # Pula arquivos vazios

                # 2. Processa cada linha
                for linha in linhas:
                    partes = linha.strip().split()
                    
                    # Verifica se a linha não está vazia e tem o formato esperado
                    if len(partes) >= 5:
                        # A MÁGICA: Substitui o ID da classe por '0'
                        partes[0] = '1'
                        
                        # Recria a linha e adiciona à nossa lista
                        novas_linhas.append(" ".join(partes) + "\n")
                        total_detecções += 1
                
                # 3. Reescreve o arquivo com as classes unificadas
                with open(caminho_arquivo, 'w') as f:
                    f.writelines(novas_linhas)
                
                arquivos_modificados += 1
            
            except Exception as e:
                print(f"Erro ao processar o arquivo {caminho_arquivo}: {e}")

    print(f"Processamento concluído.")
    print(f"Total de arquivos .txt modificados: {arquivos_modificados}")
    print(f"Total de detecções unificadas para a classe '0': {total_detecções}")
    print("---------------------------------------------------\n")

# =============================================================================
# 1. CONFIGURE SEUS CAMINHOS AQUI
# =============================================================================

# ⚠️ COLOQUE AQUI OS CAMINHOS PARA AS PASTAS DE LABELS DO SEU DATASET DE VEÍCULOS
# Lembre-se: este script MODIFICA os arquivos originais. FAÇA BACKUP.

# Exemplo: "D:/Datasets/Veiculos/train/labels"
PATH_TREINO_LABELS = "C:/Users/leonardo/Desktop/tcc/veiculos.yolov11/train\labels"

# Exemplo: "D:/Datasets/Veiculos/valid/labels"
PATH_VALID_LABELS = "C:/Users/leonardo/Desktop/tcc/veiculos.yolov11/valid/labels"

# --- NOVO ---
# Exemplo: "D:/Datasets/Veiculos/test/labels"
PATH_TEST_LABELS = "C:/Users/leonardo/Desktop/tcc/veiculos.yolov11/test/labels"


# =============================================================================
# 2. EXECUÇÃO DO SCRIPT
# =============================================================================

# String de placeholder para checagem
PLACEHOLDER = "COLOQUE_O_CAMINHO_PARA"

# Processa a pasta de treino
if PLACEHOLDER not in PATH_TREINO_LABELS:
    unificar_classes_para_zero(PATH_TREINO_LABELS)
else:
    print("AVISO: Caminho de TREINO não configurado. Pulando...")

# Processa a pasta de validação
if PLACEHOLDER not in PATH_VALID_LABELS:
    unificar_classes_para_zero(PATH_VALID_LABELS)
else:
    print("AVISO: Caminho de VALIDAÇÃO não configurado. Pulando...")

# --- NOVO ---
# Processa a pasta de teste
if PLACEHOLDER not in PATH_TEST_LABELS:
    unificar_classes_para_zero(PATH_TEST_LABELS)
else:
    print("AVISO: Caminho de TESTE não configurado. Pulando...")
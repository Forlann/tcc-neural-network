import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# --- Nossas importações customizadas ---
from YoloDataset import YoloDataset
from functions import detection_collate_fn

# --- Importações do Modelo (MODIFICADAS) ---
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# Importamos nosso backbone modificado
from CustomBackbone import CustomBackbone 

# =============================================================================
# 1. FUNÇÃO PARA CARREGAR O MODELO (MODIFICADA)
# =============================================================================

def get_detection_model(num_classes):
	"""
	Carrega um modelo de detecção Faster R-CNN usando o
	NOSSO PRÓPRIO BACKBONE customizado.
	"""

	# 1. Carrega o nosso backbone
	#    (Este agora retorna um OrderedDict([('0', tensor_saida)]) )
	backbone = CustomBackbone()
	
	# --- !! INÍCIO DA CORREÇÃO DE ARQUITETURA !! ---
	#
	# O erro (AssertionError) acontece porque o FasterRCNN padrão
	# espera um backbone com FPN (múltiplos feature maps).
	# Nosso backbone só retorna UM feature map (que chamamos de '0').
	#
	# Precisamos dizer ao FasterRCNN para usar um AnchorGenerator e um RoIPooler
	# que trabalhem com APENAS UM feature map.
	
	# 2. Cria um AnchorGenerator para UM feature map
	#    (Note os parênteses duplos, significando 1 nível de feature map)
	anchor_generator = AnchorGenerator(
		sizes=((128, 256, 512),),         # Tamanhos de âncora (pode ajustar)
		aspect_ratios=((0.5, 1.0, 2.0),) # Ratios (largo, quadrado, alto)
	)
	
	# 3. Cria um RoI Pooler para UM feature map
	#    Ele vai extrair features do map '0' (o único que nosso backbone retorna)
	roi_pooler = MultiScaleRoIAlign(
		featmap_names=['0'],
		output_size=7,       # Tamanho padrão do RoI pool
		sampling_ratio=2
	)

	# 4. Cria o modelo Faster R-CNN
	model = FasterRCNN(
		backbone,
		num_classes=91, # Padrão COCO, vamos trocar a cabeça abaixo
		rpn_anchor_generator=anchor_generator, # <-- Nosso anchor generator customizado
		box_roi_pool=roi_pooler               # <-- Nosso RoI pooler customizado
	)
	
	# 5. Substitui a "cabeça" (o classificador)
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	
	# IMPORTANTE: num_classes + 1 (para o __background__)
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

	return model

# =============================================================================
# 2. CONFIGURAÇÕES PRINCIPAIS
# =============================================================================

# --- Configure seus caminhos aqui ---
TRAIN_IMG_DIR = "D:/dev/faculdade/TCC/REDE_NEURAL/v0.2/Datasetsall/Datasetsall - Copia/train/images"
TRAIN_LBL_DIR = "D:/dev/faculdade/TCC/REDE_NEURAL/v0.2/Datasetsall/Datasetsall - Copia/train/labels"

VAL_IMG_DIR = "D:/dev/faculdade/TCC/REDE_NEURAL/v0.2/Datasetsall/Datasetsall - Copia/valid/images"
VAL_LBL_DIR = "D:/dev/faculdade/TCC/REDE_NEURAL/v0.2/Datasetsall/Datasetsall - Copia/valid/labels"

# --- Parâmetros de Treinamento ---
NUM_CLASSES = 6
BATCH_SIZE = 8
# Vamos voltar para 100 épocas, 150 pode ser muito para começar
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# =============================================================================
# 3. PREPARAÇÃO DOS DADOS (Dataset e DataLoader)
# =============================================================================

transform_train = transforms.Compose([
	transforms.Resize((416, 416)),
	transforms.ToTensor()
])

transform_val = transforms.Compose([
	transforms.Resize((416, 416)),
	transforms.ToTensor()
])

train_dataset = YoloDataset(
	images_dir=TRAIN_IMG_DIR,
	labels_dir=TRAIN_LBL_DIR,
	transform=transform_train
)

val_dataset = YoloDataset(
	images_dir=VAL_IMG_DIR,
	labels_dir=VAL_LBL_DIR,
	transform=transform_val
)

train_loader = DataLoader(
	train_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	collate_fn=detection_collate_fn
)

val_loader = DataLoader(
	val_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	collate_fn=detection_collate_fn
)

print(f"Dataset de treino carregado: {len(train_dataset)} imagens.")
print(f"Dataset de validação carregado: {len(val_dataset)} imagens.")

# =============================================================================
# 4. INICIALIZAÇÃO DO MODELO E OTIMIZADOR
# =============================================================================

model = get_detection_model(NUM_CLASSES)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================================================================
# 5. LOOP DE TREINAMENTO (LÓGICA ORIGINAL RESTAURADA)
# =============================================================================

print("🚀 Iniciando treinamento com BACKBONE CUSTOMIZADO (v2)...")

for epoch in range(NUM_EPOCHS):
	print(f"\n===== ÉPOCA {epoch+1}/{NUM_EPOCHS} =====")
	
	# --- Fase de Treino ---
	model.train()
	running_loss = 0.0

	for batch_idx, (images, targets) in enumerate(train_loader):
		
		# 1. Envia os dados para a GPU
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		# 2. Zera os gradientes
		optimizer.zero_grad()

		# 3. Forward Pass (Passa os dados pela rede)
		loss_dict = model(images, targets)

		# 4. Calcula a perda total
		loss = sum(l for l in loss_dict.values())

		# 5. Backward Pass
		loss.backward()

		# 6. Optimizer Step
		optimizer.step()

		running_loss += loss.item()

		if (batch_idx + 1) % 10 == 0:
			print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss Treino: {loss.item():.4f}")

	epoch_loss_train = running_loss / len(train_loader)
	print(f"--- Média Loss Treino Época {epoch+1}: {epoch_loss_train:.4f} ---")

	# --- Fase de Validação ---
	model.eval()
	val_loss = 0.0

	with torch.no_grad():
		for images, targets in val_loader:
			images = list(image.to(device) for image in images)
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

			# Truque para pegar a loss em modo de validação
			model.train()
			loss_dict = model(images, targets)
			model.eval()
				
			loss = sum(l for l in loss_dict.values())
			val_loss += loss.item()

	if len(val_loader) > 0:
	  epoch_loss_val = val_loss / len(val_loader)
	  print(f"--- Média Loss VALIDAÇÃO Época {epoch+1}: {epoch_loss_val:.4f} ---")
	else:
	  print("--- Validação pulada (val_loader vazio) ---")


# =============================================================================
# 6. FIM DO TREINAMENTO
# =============================================================================

print("✅ Treinamento finalizado.")

SAVEPATH = 'modelo_custom_backbone_final.pth'
torch.save(model.state_dict(), SAVEPATH)
print(f"Modelo salvo em: {SAVEPATH}")


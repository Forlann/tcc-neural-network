import os
import torch
from torch.utils.data import Dataset
from PIL import Image

def yolo_to_xyxy(yolo_box, img_width, img_height):
	"""
	Converte 1 box [x_center, y_center, w, h] (normalizado)
	para [xmin, ymin, xmax, ymax] (em pixels absolutos).
	"""
	x_center, y_center, w, h = yolo_box
	
	x_center_abs = x_center * img_width
	y_center_abs = y_center * img_height
	w_abs = w * img_width
	h_abs = h * img_height
	
	xmin = x_center_abs - (w_abs / 2)
	ymin = y_center_abs - (h_abs / 2)
	xmax = x_center_abs + (w_abs / 2)
	ymax = y_center_abs + (h_abs / 2)
	
	return [xmin, ymin, xmax, ymax]


class YoloDataset(Dataset):
	def __init__(self, images_dir, labels_dir, transform=None):
		self.images_dir = images_dir
		self.labels_dir = labels_dir
		self.transform = transform
		self.image_files = [f for f in os.listdir(
			images_dir) if f.endswith('.jpg') or f.endswith('.png')]

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		"""
		Retorna:
		- image: tensor da imagem
		- target: dicionário com 'boxes' e 'labels' no padrão torchvision
		"""
		image_file = self.image_files[idx]
		label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')

		image_path = os.path.join(self.images_dir, image_file)
		image = Image.open(image_path).convert('RGB')
		
		# Pega o tamanho ORIGINAL da imagem antes de qualquer transformação
		w_orig, h_orig = image.size

		label_path = os.path.join(self.labels_dir, label_file)
		
		boxes = []
		labels = []

		if os.path.exists(label_path):
			with open(label_path, 'r') as f:
				for line in f.readlines():
					# Lê os dados do YOLO
					# Tenta ler, mas pula linhas mal formatadas
					try:
						class_id, x_center, y_center, width, height = map(
							float, line.strip().split())
					except ValueError:
						print(f"WARN: Linha mal formatada pulada no arquivo: {label_file}")
						continue
					
					yolo_box = [x_center, y_center, width, height]
					
					# 1. CONVERTE A BOX para [xmin, ymin, xmax, ymax] em pixels
					pascal_box = yolo_to_xyxy(yolo_box, w_orig, h_orig)
					boxes.append(pascal_box)
					
					# 2. AJUSTA A CLASSE: +1 (porque 0 é fundo)
					# IMPORTANTE: Seu script espera 11 classes, então os
					# class_id's no seu .txt devem ir de 0 a 10.
					# O modelo receberá de 1 a 11.
					labels.append(int(class_id) + 1)

		# --- INÍCIO DA CORREÇÃO ---
		# Aqui está a correção para o erro torch.Size([0])
		
		if len(boxes) > 0:
			boxes = torch.tensor(boxes, dtype=torch.float32)
		else:
			# Se 'boxes' for uma lista vazia [], criamos um tensor
			# com a forma correta [0, 4] que o modelo espera.
			boxes = torch.empty((0, 4), dtype=torch.float32)
			
		# --- FIM DA CORREÇÃO ---
			
		labels = torch.tensor(labels, dtype=torch.int64)

		# 3. CRIA O DICIONÁRIO 'target'
		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = torch.tensor([idx]) # ID da imagem
		
		# (Opcional, mas bom para o modelo) Calcula a área das boxes
		# Este código já estava correto e lidava bem com o tensor [0, 4]
		if boxes.shape[0] > 0:
			area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		else:
			# Se não houver boxes, precisa ser um tensor vazio com shape (0,)
			area = torch.empty((0,), dtype=torch.float32)
			
		target["area"] = area
		
		# (Opcional, mas bom para o modelo) Assume que não há "multidões"
		target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
		
		# 4. APLICA A TRANSFORMAÇÃO (APENAS NA IMAGEM)
		# O modelo Faster R-CNN é inteligente o suficiente para
		# ajustar as 'targets' ao tamanho da imagem transformada.
		if self.transform:
			image = self.transform(image)
			
		# 5. RETORNA O NOVO FORMATO
		return image, target

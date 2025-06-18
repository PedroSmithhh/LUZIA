import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Obter a raiz do projeto
root_path = Path(__file__).parent.parent  # Volta uma pasta acima para encontrar a raiz

# Definir o caminho para o dataset IDRiD
base_dir = root_path / "data" / "Segmentation"
img_dir = os.path.join(base_dir, "Original Images", "Training Set")  # Caminho da pasta com imagens de treino
mask_dirs = {  # Caminho das pastas com as máscaras de cada lesão
    "MA": os.path.join(base_dir, "Segmentation Groundtruths", "Training Set", "Microaneurysms"),
    "HE": os.path.join(base_dir, "Segmentation Groundtruths", "Training Set", "Haemorrhages"),
    "EX": os.path.join(base_dir, "Segmentation Groundtruths", "Training Set", "Hard Exudates"),
    "SE": os.path.join(base_dir, "Segmentation Groundtruths", "Training Set", "Soft Exudates"),
    "OD": os.path.join(base_dir, "Segmentation Groundtruths", "Training Set", "Optic Disc")
}

# Função para carregar e pré-processar uma imagem e suas máscaras
def load_and_preprocess(img_path, mask_paths, target_size=(256, 256)):
    # Carregar a imagem de retina
    img = cv2.imread(img_path)  # Lê a imagem em formato BGR (padrão da biblioteca cv2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para RGB
    img = cv2.resize(img, target_size)  # Redimensiona para 256x256
    img = img / 255.0  # Normaliza os valores dos pixels para [0, 1]
    
    # Carregar as máscaras para cada tipo de lesão
    masks = []
    for lesion in ["MA", "HE", "EX", "SE", "OD"]:
        mask_path = mask_paths[lesion]  # Caminho da máscara para a lesão atual
        if os.path.exists(mask_path):  # Verifica se a máscara existe
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Carrega em escala de cinza (De 3 canais (RGB) vai pra 1 canal (cinza))
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)  # Redimensiona, preservando valores discretos
            mask = (mask > 0).astype(np.uint8)  # Binariza: pixels > 0 viram 1, outros 0. uint8 -> compatibilidade com Numpy
        else:
            mask = np.zeros(target_size, dtype=np.uint8)  # Cria máscara zerada para lesões ausentes
        masks.append(mask)
    
    # Empilhar as máscaras em um único array
    masks = np.stack(masks, axis=-1)  # Forma: (256, 256, 5), um canal por lesão
    
    # Criar um array de rótulos para cada pixel
    # 0 = fundo, 1 = MA, 2 = HE, 3 = EX, 4 = SE, 5 = OD
    label_map = np.zeros(target_size, dtype=np.uint8)  # Começa com 0 (fundo)
    for i, lesion in enumerate(["MA", "HE", "EX", "SE", "OD"]):
        label_map[masks[:, :, i] == 1] = i + 1  # Atribui valores 1, 2, 3, 4, 5 para cada lesão
        #plt.imshow(label_map)
    
    # Converter para one-hot encoding
    # Cada pixel vira um vetor de 6 elementos (ex.: [1, 0, 0, 0, 0, 0] para fundo)
    masks = tf.keras.utils.to_categorical(label_map, num_classes=6)  # Forma: (256, 256, 6)
    #np.set_printoptions(threshold=np.inf)
    #print(masks)
    
    return img, masks

# Função para salvar os dados pré-processados
def save_preprocessed_data(images, masks, save_dir):
    # Criar diretório para salvar os dados, se não existir
    os.makedirs(save_dir, exist_ok=True)
    
    # Salvar imagens e máscaras como arquivos .npy
    np.save(os.path.join(save_dir, "images.npy"), images)  # Salva array de imagens
    np.save(os.path.join(save_dir, "masks.npy"), masks)    # Salva array de máscaras
    print(f"Dados salvos em: {save_dir}")

# Função para carregar os dados pré-processados
def load_preprocessed_data(load_dir="caminho/para/preprocessed_data"):
    # Carregar os arrays salvos
    images = np.load(os.path.join(load_dir, "images.npy"))
    masks = np.load(os.path.join(load_dir, "masks.npy"))
    print("Dados carregados!")
    print("Formato das imagens:", images.shape)
    print("Formato das máscaras:", masks.shape)
    return images, masks

# Carregar todas as imagens e máscaras
images = []
masks = []
for img_file in os.listdir(img_dir):
    if img_file.endswith(".jpg"):  # Verifica se é uma imagem JPG
        img_path = os.path.join(img_dir, img_file)  # Caminho completo da imagem
        # Mapear os caminhos das máscaras correspondentes
        mask_paths = {
            "MA": os.path.join(mask_dirs["MA"], img_file.replace(".jpg", "_MA.tif")),
            "HE": os.path.join(mask_dirs["HE"], img_file.replace(".jpg", "_HE.tif")),
            "EX": os.path.join(mask_dirs["EX"], img_file.replace(".jpg", "_EX.tif")),
            "SE": os.path.join(mask_dirs["SE"], img_file.replace(".jpg", "_SE.tif")),
            "OD": os.path.join(mask_dirs["OD"], img_file.replace(".jpg", "_OD.tif"))
        }
        # Chamar a função de pré-processamento
        img, mask = load_and_preprocess(img_path, mask_paths)
        images.append(img)  # Adicionar imagem à lista
        masks.append(mask)  # Adicionar máscara à lista

# Converter listas para arrays NumPy
images = np.array(images)  # Forma: (n_imagens, 256, 256, 3)
masks = np.array(masks)    # Forma: (n_imagens, 256, 256, 6)

# Exibir o formato dos dados
print("Formato das imagens:", images.shape)  # Exemplo: (54, 256, 256, 3)
print("Formato das máscaras:", masks.shape)  # Exemplo: (54, 256, 256, 6)

# Salvar os dados pré-processados
save_dir = str(root_path / "preprocessing" / "preprocessed_data")  # Define o diretório para salvar
save_preprocessed_data(images, masks, save_dir)

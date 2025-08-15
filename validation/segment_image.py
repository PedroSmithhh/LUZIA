import os
import sys
import argparse
import logging
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_NAMES = {
    0: 'Fundo',
    1: 'Microaneurisma (MA)',
    2: 'Hemorragia (HE)',
    3: 'Exsudato Duro (EX)',
    4: 'Exsudato Mole (SE)',
    5: 'Disco Óptico (OD)'
}

CLASS_COLORS = {
    0: [0, 0, 0],        # Preto (Fundo)
    1: [255, 0, 0],      # Vermelho (MA)
    2: [0, 255, 0],      # Verde (HE)
    3: [0, 0, 255],      # Azul (EX)
    4: [255, 255, 0],    # Amarelo (SE)
    5: [255, 0, 255]     # Magenta (OD)
}

def setup_logging():
    """Configura o sistema de logging para exibir mensagens no console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def load_model(model_path: str) -> keras.Model:
    """Carrega o modelo de segmentação treinado."""
    logging.info(f"Carregando modelo de: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Arquivo de modelo não encontrado em: {model_path}")
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = keras.models.load_model(model_path, compile=False)
    logging.info("Modelo carregado com sucesso.")
    return model

def preprocess_image(image_path: str, target_size: tuple = (256, 256)):
    """Carrega e pré-processa a imagem de entrada."""
    logging.info(f"Processando imagem: {os.path.basename(image_path)}")
    if not os.path.exists(image_path):
        logging.error(f"Arquivo de imagem não encontrado em: {image_path}")
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Falha ao carregar a imagem: {image_path}")
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image_rgb.shape[:2]

    image_resized = cv2.resize(image_rgb, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    logging.info(f"Imagem pré-processada: {original_shape} -> {target_size}")
    return image_batch, image_rgb, original_shape

def predict_mask(model: keras.Model, image_batch: np.ndarray) -> np.ndarray:
    """Realiza a predição da máscara de segmentação."""
    logging.info("Realizando predição da máscara...")
    prediction = model.predict(image_batch, verbose=0)
    mask = np.argmax(prediction[0], axis=-1)
    logging.info("Predição concluída.")
    return mask

def create_colored_mask(mask: np.ndarray) -> np.ndarray:
    """Cria uma máscara colorida a partir dos índices de classe."""
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored_mask[mask == class_id] = color
    return colored_mask

def create_overlay(original_image: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.6) -> tuple:
    """Sobrepõe a máscara colorida na imagem original."""
    original_h, original_w = original_image.shape[:2]
    mask_resized = cv2.resize(colored_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(original_image, 1 - alpha, mask_resized, alpha, 0)
    return overlay, mask_resized

def save_results(original_image, colored_mask, overlay, output_dir, image_name):
    """Salva as imagens resultantes (original, máscara e sobreposição)."""
    logging.info(f"Salvando resultados no diretório: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]

    paths = {
        "original": os.path.join(output_dir, f"{base_name}_original.png"),
        "mask": os.path.join(output_dir, f"{base_name}_mask.png"),
        "overlay": os.path.join(output_dir, f"{base_name}_overlay.png")
    }

    # Legenda comum para máscara e overlay
    legend_elements = [mpatches.Patch(color=np.array(color)/255.0, label=name)
                       for name, color in zip(CLASS_NAMES.values(), CLASS_COLORS.values())]

    # Salvar imagens
    plt.imsave(paths["original"], original_image)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(colored_mask)
    ax.set_title("Máscara de Segmentação")
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(paths["mask"], bbox_inches='tight', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title("Sobreposição: Imagem + Segmentação")
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(paths["overlay"], bbox_inches='tight', dpi=150)
    plt.close(fig)

    logging.info(f"Imagem original salva em: {paths['original']}")
    logging.info(f"Máscara de segmentação salva em: {paths['mask']}")
    logging.info(f"Imagem de sobreposição salva em: {paths['overlay']}")
    return paths

def analyze_prediction(mask: np.ndarray):
    """Calcula e loga a distribuição de pixels por classe na máscara prevista."""
    logging.info("Análise da Segmentação:")
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    analysis_results = "\n" + "-" * 50
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        class_name = CLASS_NAMES.get(class_id, f"Classe Desconhecida {class_id}")
        analysis_results += f"\n{class_name:20}: {count:8d} pixels ({percentage:5.2f}%)"
    analysis_results += "\n" + "-" * 50
    logging.info(analysis_results)

def main():
    """Função principal para orquestrar o processo de segmentação."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Segmentação de lesões em imagens de retina.")
    parser.add_argument("image_path", help="Caminho para a imagem a ser segmentada.")
    parser.add_argument("--model", default="unet_model.keras", help="Caminho para o arquivo do modelo treinado (.keras).")
    parser.add_argument("--output", "-o", default="outputs", help="Diretório de saída para os resultados.")
    
    args = parser.parse_args()
    
    try:
        logging.info("Iniciando processo de segmentação de imagem...")
        
        model = load_model(args.model)
        image_batch, original_image, _ = preprocess_image(args.image_path)
        mask = predict_mask(model, image_batch)
        
        # Redimensionar a máscara prevista para o tamanho 256x256 antes de criar a versão colorida
        colored_mask_256 = create_colored_mask(mask)
        overlay, resized_colored_mask = create_overlay(original_image, colored_mask_256)
        
        image_name = os.path.basename(args.image_path)
        save_results(original_image, resized_colored_mask, overlay, args.output, image_name)
        
        analyze_prediction(mask)
        
        logging.info(f"Segmentação concluída com sucesso. Resultados salvos em: {os.path.abspath(args.output)}")
        
    except FileNotFoundError as e:
        logging.error(f"Erro de arquivo: {e}. Verifique os caminhos fornecidos.")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Ocorreu um erro inesperado durante a execução: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
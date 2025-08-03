#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para segmentação de imagens usando modelo U-Net treinado
Salva a imagem original, máscara predita e overlay colorido
"""

import os
import sys
import argparse
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Dicionário de classes e cores
CLASS_NAMES = {
    0: 'Background',
    1: 'Microaneurisma (MA)',
    2: 'Hemorragia (HE)', 
    3: 'Exsudato Duro (EX)',
    4: 'Exsudato Mole (SE)',
    5: 'Disco Óptico (OD)'
}

# Cores para cada classe (RGB)
CLASS_COLORS = {
    0: [0, 0, 0],        # Preto - Background
    1: [255, 0, 0],      # Vermelho - Microaneurisma
    2: [0, 255, 0],      # Verde - Hemorragia
    3: [0, 0, 255],      # Azul - Exsudato Duro
    4: [255, 255, 0],    # Amarelo - Exsudato Mole
    5: [255, 0, 255]     # Magenta - Disco Óptico
}

def load_model():
    """Carrega o modelo treinado"""
    model_path = 'unet_model.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    print("🔄 Carregando modelo...")
    model = keras.models.load_model(model_path, compile=False)
    print("✅ Modelo carregado com sucesso!")
    return model

def preprocess_image(image_path):
    """Carrega e preprocessa a imagem"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    print(f"🔄 Carregando imagem: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    # Converter BGR para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image_rgb.shape[:2]
    
    # Redimensionar para 256x256 (tamanho usado no treinamento)
    image_resized = cv2.resize(image_rgb, (256, 256))
    
    # Normalizar para [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Adicionar dimensão do batch
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    print(f"✅ Imagem preprocessada: {original_shape} -> (256, 256)")
    return image_batch, image_rgb, original_shape

def predict_mask(model, image_batch):
    """Faz a predição da máscara"""
    print("🔄 Fazendo predição...")
    prediction = model.predict(image_batch, verbose=0)
    
    # Converter de one-hot para classes
    mask = np.argmax(prediction[0], axis=-1)
    
    print("✅ Predição concluída!")
    return mask

def create_colored_mask(mask):
    """Cria máscara colorida baseada nas classes"""
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        colored_mask[mask == class_id] = color
    
    return colored_mask

def create_overlay(original_image, colored_mask, alpha=0.6):
    """Cria overlay da máscara sobre a imagem original"""
    # Redimensionar máscara para o tamanho original
    original_h, original_w = original_image.shape[:2]
    mask_resized = cv2.resize(colored_mask, (original_w, original_h))
    
    # Criar overlay
    overlay = cv2.addWeighted(original_image, 1-alpha, mask_resized, alpha, 0)
    
    return overlay, mask_resized

def save_results(original_image, colored_mask, overlay, output_dir, image_name):
    """Salva os resultados"""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(image_name)[0]
    
    # Salvar imagem original
    original_path = os.path.join(output_dir, f"{base_name}_original.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.title("Imagem Original")
    plt.axis('off')
    plt.savefig(original_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Salvar máscara colorida
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(colored_mask)
    plt.title("Máscara de Segmentação")
    plt.axis('off')
    
    # Criar legenda
    legend_elements = [mpatches.Patch(color=np.array(color)/255.0, label=name) 
                      for class_id, (color, name) in enumerate(zip(CLASS_COLORS.values(), CLASS_NAMES.values()))]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(mask_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Salvar overlay
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title("Sobreposição: Imagem + Segmentação")
    plt.axis('off')
    
    # Adicionar legenda
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(overlay_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return original_path, mask_path, overlay_path

def analyze_prediction(mask):
    """Analisa a predição e mostra estatísticas"""
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    print("\n📊 Análise da Segmentação:")
    print("-" * 50)
    
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        class_name = CLASS_NAMES.get(class_id, f"Classe {class_id}")
        print(f"{class_name:20}: {count:6d} pixels ({percentage:5.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Segmentação de imagem de retina')
    parser.add_argument('image_path', help='Caminho para a imagem a ser segmentada')
    parser.add_argument('--output', '-o', default='outputs', 
                       help='Diretório de saída (padrão: outputs)')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Iniciando segmentação de imagem...")
        print("=" * 60)
        
        # Carregar modelo
        model = load_model()
        
        # Preprocessar imagem
        image_batch, original_image, original_shape = preprocess_image(args.image_path)
        
        # Fazer predição
        mask = predict_mask(model, image_batch)
        
        # Criar máscara colorida
        colored_mask = create_colored_mask(mask)
        
        # Criar overlay
        overlay, mask_resized = create_overlay(original_image, colored_mask)
        
        # Salvar resultados
        image_name = os.path.basename(args.image_path)
        original_path, mask_path, overlay_path = save_results(
            original_image, mask_resized, overlay, args.output, image_name
        )
        
        # Analisar predição
        analyze_prediction(mask)
        
        print("\n💾 Arquivos salvos:")
        print(f"   📷 Original: {original_path}")
        print(f"   🎨 Máscara:  {mask_path}")
        print(f"   🔄 Overlay:  {overlay_path}")
        
        print(f"\n✅ Segmentação concluída com sucesso!")
        print(f"📁 Resultados salvos em: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import numpy as np
import os

def analyze_dataset_distribution():
    """
    Analisa a distribuição do dataset de duas formas:
    1. Porcentagem de pixels que cada classe ocupa (densidade)
    2. Porcentagem de imagens onde cada classe aparece (presença)
    """
    
    # Carregar dados preprocessados
    masks_path = '../preprocessing/preprocessed_data/masks.npy'
    masks = np.load(masks_path)
    
    print("=== ANÁLISE DA DISTRIBUIÇÃO DO DATASET IDRiD ===\n")
    
    # Converter de one-hot para labels
    label_masks = np.argmax(masks, axis=-1)  # Shape: (54, 256, 256)
    
    num_images = masks.shape[0]
    total_pixels = masks.size // 6  # Total de pixels em todas as imagens
    
    classes = {
        0: "Fundo",
        1: "Microaneurismas (MA)", 
        2: "Hemorragias (HE)",
        3: "Exsudatos Duros (EX)",
        4: "Exsudatos Moles (SE)",
        5: "Disco Óptico (OD)"
    }
    
    print("1. DISTRIBUIÇÃO POR PIXELS (Densidade das Classes)")
    print("-" * 60)
    for class_id in range(6):
        class_pixels = np.sum(masks[:,:,:,class_id])
        pixel_percentage = (class_pixels / total_pixels) * 100
        print(f"Classe {class_id} ({classes[class_id]}): {pixel_percentage:.2f}% dos pixels")
    
    print("\n2. DISTRIBUIÇÃO POR IMAGENS (Presença das Classes)")
    print("-" * 60)
    for class_id in range(6):
        # Contar em quantas imagens a classe aparece
        images_with_class = 0
        for img_idx in range(num_images):
            if np.any(label_masks[img_idx] == class_id):
                images_with_class += 1
        
        image_percentage = (images_with_class / num_images) * 100
        print(f"Classe {class_id} ({classes[class_id]}): aparece em {images_with_class}/{num_images} imagens ({image_percentage:.1f}%)")
    
    print("\n3. ANÁLISE DETALHADA POR CLASSE")
    print("-" * 60)
    for class_id in range(6):
        class_pixels_per_image = []
        images_with_class = 0
        
        for img_idx in range(num_images):
            class_pixels_in_image = np.sum(label_masks[img_idx] == class_id)
            total_pixels_in_image = label_masks[img_idx].size
            
            if class_pixels_in_image > 0:
                images_with_class += 1
                percentage_in_image = (class_pixels_in_image / total_pixels_in_image) * 100
                class_pixels_per_image.append(percentage_in_image)
        
        if class_pixels_per_image:
            avg_percentage = np.mean(class_pixels_per_image)
            max_percentage = np.max(class_pixels_per_image)
            min_percentage = np.min(class_pixels_per_image)
            
            print(f"\n{classes[class_id]}:")
            print(f"  - Presente em: {images_with_class}/{num_images} imagens ({(images_with_class/num_images)*100:.1f}%)")
            print(f"  - Quando presente, ocupa em média: {avg_percentage:.2f}% da imagem")
            print(f"  - Variação: {min_percentage:.2f}% - {max_percentage:.2f}% da imagem")
        else:
            print(f"\n{classes[class_id]}: Não encontrada em nenhuma imagem")

if __name__ == "__main__":
    analyze_dataset_distribution()

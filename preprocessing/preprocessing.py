import os
import shutil
import random
import tensorflow as tf

def organizar_dataset(original_dir, base_dir, split_size=0.8):
    """
    Organiza o dataset de imagens em pastas de treino e validação.
    """
    train_dir = os.path.join(base_dir, 'treino')
    validation_dir = os.path.join(base_dir, 'validacao')

    # Cria a pasta base e as subpastas se não existirem
    if os.path.exists(base_dir):
        print(f"A pasta de destino '{base_dir}' já existe. Removendo-a para começar do zero.")
        shutil.rmtree(base_dir)

    os.makedirs(train_dir)
    os.makedirs(validation_dir)
    print(f"Estrutura de pastas criada em: '{base_dir}'")

    for class_name in os.listdir(original_dir):
        class_path = os.path.join(original_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # Criar subpastas de classe nos diretórios de treino e validação
        os.makedirs(os.path.join(train_dir, class_name))
        os.makedirs(os.path.join(validation_dir, class_name))

        # Listar todas as imagens da classe e embaralhá-las
        all_files = os.listdir(class_path)
        random.shuffle(all_files)

        # Calcular o ponto de divisão
        split_point = int(len(all_files) * split_size)
        train_files = all_files[:split_point]
        validation_files = all_files[split_point:]

        # Copiar arquivos para as pastas de destino
        for file_name in train_files:
            source_file = os.path.join(class_path, file_name)
            dest_file = os.path.join(train_dir, class_name, file_name)
            shutil.copyfile(source_file, dest_file)

        for file_name in validation_files:
            source_file = os.path.join(class_path, file_name)
            dest_file = os.path.join(validation_dir, class_name, file_name)
            shutil.copyfile(source_file, dest_file)

        print(f" -> {len(train_files)} imagens copiadas para treino.")
        print(f" -> {len(validation_files)} imagens copiadas para validação.")

    print("\nOrganização do dataset concluída com sucesso!")

ORIGINAL_DATASET_DIR = 'data/'
NEW_BASE_DIR = 'dataset/'

organizar_dataset(ORIGINAL_DATASET_DIR, NEW_BASE_DIR, split_size=0.8)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_dir = os.path.join(NEW_BASE_DIR, 'treino')
validation_dir = os.path.join(NEW_BASE_DIR, 'validacao')

# Data Augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Gerador para os dados de validação
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True, # Embaralha os dados de treino
    seed=SEED
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # Não embaralha os dados de validação
)
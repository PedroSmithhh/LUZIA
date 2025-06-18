import tensorflow as tf
from tf_keras import layers, models

def unet_model(input_size=(256, 256, 3), num_classes=6):
    # Entrada do modelo
    inputs = layers.Input(input_size)
    
    # Encoder (parte de contração)
    # Camada 1
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)  # Reduz para 128x128
    
    # Camada 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)  # Reduz para 64x64
    
    # Camada 3 (bottleneck)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder (parte de expansão)
    # Camada 4
    u4 = layers.UpSampling2D((2, 2))(c3)  # Aumenta para 128x128
    u4 = layers.concatenate([u4, c2])  # Conexão de salto com c2
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    # Camada 5
    u5 = layers.UpSampling2D((2, 2))(c4)  # Aumenta para 256x256
    u5 = layers.concatenate([u5, c1])  # Conexão de salto com c1
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    # Camada de saída
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)
    
    # Criar o modelo
    model = models.Model(inputs, outputs)
    return model

# Criar e compilar o modelo
model = unet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()  # Mostra a arquitetura do modelo
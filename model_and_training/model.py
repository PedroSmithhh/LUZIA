import tensorflow as tf
import keras
from keras import layers, models
import keras.ops as ops 

def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Métrica Dice Coefficient. Mede a sobreposição entre a máscara prevista e a real.
    Ótima para segmentação e lida bem com desbalanceamento.
    """
    y_true_f = ops.reshape(y_true, [-1])
    y_pred_f = ops.reshape(y_pred, [-1])

    intersection = ops.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (ops.sum(y_true_f) + ops.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Função de perda baseada no Dice Coefficient.
    Minimizar esta perda é o mesmo que maximizar o Dice Coefficient.
    """
    return 1 - dice_coefficient(y_true, y_pred)

def unet_model(input_size=(256, 256, 3), num_classes=6):
    inputs = layers.Input(input_size)
    
    # --- Encoder (Parte de Contração) ---

    # Camada 1
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.25)(p1) # Desliga 25% dos neurônios para evitar overfitting
    
    # Camada 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2) 
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.5)(p2) # Dropout maior em camadas mais profundas
    
    # Camada 3 (Bottleneck)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    
    # --- Decoder (Parte de Expansão) ---
    
    # Camada 4
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.concatenate([u4, c2]) # Conexão de salto com c2
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    
    # Camada 5
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c1]) # Conexão de salto com c1
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    
    # Camada de saída
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)
    
    # Criar o modelo
    model = models.Model(inputs, outputs)
    return model

# Criar e compilar o modelo
model = unet_model()
model.compile(optimizer='adam', 
              loss=dice_loss,
              metrics=[dice_coefficient])
model.summary()  # Mostra a arquitetura do modelo
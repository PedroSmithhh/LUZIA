from sklearn.model_selection import train_test_split
from model import model
import os
import numpy as np
from pathlib import Path
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Definição dos caminhos
images_path = os.path.join(base_dir, 'preprocessing', 'preprocessed_data', 'images.npy')
masks_path = os.path.join(base_dir, 'preprocessing', 'preprocessed_data', 'masks.npy')

# Salva o modelo com o melhor 'val_dice_coefficient' visto até o momento
model_checkpoint = ModelCheckpoint('unet_model_best.keras', 
                                   monitor='val_dice_coefficient', 
                                   save_best_only=True, 
                                   mode='max', # Maximizar o dice
                                   verbose=1)

# Reduz a taxa de aprendizado se a métrica de validação estagnar
reduce_lr = ReduceLROnPlateau(monitor='val_dice_coefficient', 
                              factor=0.1, 
                              patience=5, 
                              mode='max',
                              min_lr=1e-6,
                              verbose=1)

# Carregar os arquivos
images = np.load(images_path)
masks = np.load(masks_path)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# Treinar o modelo
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=50, 
                    batch_size=8,
                    verbose=1,
                    callbacks=[model_checkpoint, reduce_lr]
                    )

# Salvar o modelo final de qualquer maneira
model.save('unet_model_final.keras')
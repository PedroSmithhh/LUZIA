from sklearn.model_selection import train_test_split
from model import model
import os
import numpy as np
from pathlib import Path

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

images_path = os.path.join(base_dir, 'preprocessing', 'preprocessed_data', 'images.npy')
masks_path = os.path.join(base_dir, 'preprocessing', 'preprocessed_data', 'masks.npy')

# Carregar os arquivos
images = np.load(images_path)
masks = np.load(masks_path)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# Treinar o modelo
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=20, 
                    batch_size=8,
                    verbose=1)

# Salvar o modelo
model.save('unet_model.keras')
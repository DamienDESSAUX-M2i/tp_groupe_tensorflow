import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def create_baseline_cnn(data_augmentation, img_size):
    model_layers = []
    model_layers.append(data_augmentation)
    model_layers.extend([
        layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(img_size, img_size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),

        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(100, activation='softmax')
    ])

    return keras.Sequential(model_layers)



def create_baseline_resnet(data_augmentation, img_size):
    model_layers = []
    model_layers.append(data_augmentation)
    model_layers.extend([
        layers.Conv2D(8, 3, activation= "relu", input_shape = (img_size, img_size, 3), padding = "same", name = "conv1"),
        layers.BatchNormalization(),

        layers.Conv2D(16, 3, activation= "relu", padding = "same", name = "conv2"),
        layers.BatchNormalization(),

        layers.Conv2D(32, 3, activation= "relu", padding = "same", name = "conv3"),
        layers.BatchNormalization(),

        layers.Conv2D(64, 3, activation= "relu", padding = "same", name = "conv4"),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, activation= "relu", padding = "same", name = "conv5"),
        layers.BatchNormalization(),

        layers.Conv2D(256, 3, activation= "relu", padding = "same", name = "conv6"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(100, activation='softmax')

    ])
    return keras.Sequential(model_layers)



def create_resnet_custom(data_augmentation, img_size):
    inputs = keras.Input(shape=(img_size, img_size, 3))

    x = data_augmentation(inputs)

    # --- BLOC 1 & 2 ---
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # On garde une trace de x pour le skip
    shortcut = x 
    
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # SKIP CONNECTION 1 (Addition)
    x = layers.Add()([x, shortcut]) 

    # --- BLOC 3 & 4 (Changement de dimension) ---
    # Quand on passe de 8 à 32 filtres, on doit adapter le shortcut
    shortcut = layers.Conv2D(32, 1, padding="same")(x) 
    
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # SKIP CONNECTION 2
    x = layers.Add()([x, shortcut])

    # --- SORTIE ---
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(100, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model



def create_mobileNetV2(data_augmentation, img_size):
    model_layers = []
    model_layers.append(data_augmentation)

    base_model = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), 
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model_layers.extend([
        layers.Resizing(img_size, img_size),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(100, activation='softmax') 
    ])
    return keras.Sequential(model_layers)



def create_efficientNetB0(data_augmentation, img_size):
    model_layers = []
    model_layers.append(data_augmentation)

    base_model = keras.applications.EfficientNetB0(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)

    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model_layers.extend([
        layers.Resizing(img_size, img_size),  # Redimensionner à l'entrée
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(100, activation='softmax')  # 100 classes!
    ])
    return keras.Sequential(model_layers)



def create_hierarchical_model(data_augmentation, img_size):
    inputs = keras.Input(shape=(img_size, img_size, 3))

    x = data_augmentation(inputs)

    # --- TRONC COMMUN (Extraction de caractéristiques) ---
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x) # Vecteur de caractéristiques riches

    # --- BRANCHE 1 : Super-classes (20 classes) ---
    # On aide le modèle à apprendre les concepts larges
    coarse_branch = layers.Dense(256, activation="relu")(x)
    coarse_branch = layers.Dropout(0.3)(coarse_branch)
    output_coarse = layers.Dense(20, activation="softmax", name="coarse_output")(coarse_branch)

    # --- BRANCHE 2 : Classes fines (100 classes) ---
    # On concatène parfois la sortie coarse pour aider la branche fine
    combined = layers.Concatenate()([x, coarse_branch]) 
    fine_branch = layers.Dense(512, activation="relu")(combined)
    fine_branch = layers.Dropout(0.4)(fine_branch)
    output_fine = layers.Dense(100, activation="softmax", name="fine_output")(fine_branch)

    # Création du modèle avec 1 entrée et 2 sorties
    model = keras.Model(inputs=inputs, outputs=[output_coarse, output_fine])
    return model




def create_model(data_augmentation, img_size, nb_conv2D = 4, start_filters = 32):
    my_layers = []
    my_layers.append(layers.Input(shape = (img_size, img_size, 3)))
    my_layers.append(data_augmentation)

    filters = start_filters
    for i in range(nb_conv2D):
        my_layers.append(layers.Conv2D(filters, 3, activation="relu", padding="same"))
        my_layers.append(layers.BatchNormalization())
        if i % 2 == 1:
            my_layers.append(layers.MaxPooling2D(2))
            filters = filters * 2

    my_layers.extend([
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(100, activation="softmax")
    ])

    return keras.Sequential(my_layers)




if __name__ == "__main__":
    img_size = 32 
    # augmentation vide pour le test si besoin
    dummy_aug = tf.keras.Sequential([layers.Layer()]) 

    # Initialisation
    model = create_hierarchical_model(dummy_aug, img_size)
    model.summary()

    # Simulation d'une image (Batch de 1)
    test_input = np.random.random((1, img_size, img_size, 3)).astype(np.float32)
    predictions = model.predict(test_input)
    
    print("--- Résultats du Test ---")
    if isinstance(predictions, list):
        print(f" Sorties multiples détectées : {len(predictions)}")
        print(f"   1. Super-classes (Coarse) : {predictions[0].shape}") 
        print(f"   2. Classes fines (Fine)    : {predictions[1].shape}")  
    else:
        print(f" Forme de la sortie : {predictions.shape}")
    
    print("Test de prédiction réussi !")
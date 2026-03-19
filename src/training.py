import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import time 

### Phase 4 : Entraînement & Optimisation

def compile_model(model, optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]):
    """"
    Compiler un modèle.
    
    [args]
        model (keras.Model): Le modèle à compiler.
        optimizer (str): L'optimisateur, par defaut "adam".
        loss (str): La fonction de perte, par defaut "categorical_crossentropy".
        metrics (list): La liste des métriques, par defaut ["accuracy"].
        
    returns
        model (keras.Model): Le modèle compilé.
    """
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

def train_model(model, train_ds, train_val, callbacks, epochs=1):
    """"
    Entrainer un modèle sur les données d'entrainement et de validation.
    Retourne l'historique d'entraînement.
    
    [args]
        model (keras.Model): Le modèle à entrainer.
        X_train (np.ndarray): Les données d'entrainement.
        X_val (np.ndarray): Les données de validation.
        callbacks (list): La liste des callbacks.
        epochs (int): Le nombre d'epochs, par defaut 1.
        
    returns
        history (keras.callbacks.History): L'historique d'entraînement.
    """
    history = model.fit(
        train_ds,
        validation_data=train_val,
        callbacks=callbacks,
        epochs=epochs,
        verbose=1
    )
    return history
    
def plot_history(history):
    """"
    Visualiser l'historique d'entraînement.
    
    [args]
        history (keras.callbacks.History): L'historique d'entraînement.
        
    returns
        None
    """
    hist = history.history
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LOSS
    axes[0].plot(hist["loss"], label="train loss")
    axes[0].plot(hist["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # ACCURACY
    axes[1].plot(hist["accuracy"], label="train accuracy")
    axes[1].plot(hist["val_accuracy"], label="val accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    

def export_history_csv(history, model_name="model", save_dir="src/history"):
    """"
    exporter l'historique d'entraînement en CSV.
    
    [args]
        history (keras.callbacks.History): L'historique d'entraînement.
        model_name (str): Le nom du modèle.
        save_dir (str): Le dossier de sauvegarde.
    """
    os.makedirs(save_dir, exist_ok=True)

    history_df = pd.DataFrame(history.history)
    path = os.path.join(save_dir, f"{model_name}_history.csv")

    history_df.to_csv(path, index=False)
    print(f"CSV saved at: {path}")
    

def export_history_json(history, model_name="model", save_dir="src/history"):
    """"
    exporter l'historique d'entraînement en json.
    
    [args]
        history (keras.callbacks.History): L'historique d'entraînement.
        model_name (str): Le nom du modèle.
        save_dir (str): Le dossier de sauvegarde.
    """
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"{model_name}_history.json")

    with open(path, "w") as f:
        json.dump(history.history, f)

    print(f"JSON saved at: {path}")


# ---------------------------------------------------------------------

# ## Phase 5 : Évaluation Complète

def compute_metrics(model, test_ds):
    """
    Calculer les métriques globales.
    
    [args]
        model (keras.Model): Le modèle à tester.
        test_ds (tf.data.Dataset): Le jeu de données de test.
        
    returns
        y_true (np.ndarray): Les vraies classes.
        y_pred (np.ndarray): Les predictions.
        y_pred_probs (np.ndarray): Les probabilités de predictions.
    """
    # TODO
    #_true = np.concatenate([y.numpy() for _, y in test_ds.unbatch()])
    y_true = np.array([y.numpy() for _, y in test_ds.unbatch()])

    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return y_true, y_pred, y_pred_probs

def compute_accuracy(y_true, y_pred, y_pred_probs, top_k=3):
    """
    Calculer l'accuracy et le top-k accuracy.
    
    [args]
        y_true (np.ndarray): Les vraies classes.
        y_pred (np.ndarray): Les predictions.
        y_pred_probs (np.ndarray): Les probabilités de predictions.
        top_k (int): Le top-k accuracy, par defaut 3.
        
    returns
        acc (float): L'accuracy.
        top_k_acc (float): Le top-k accuracy.
    """

    # Accuracy classique
    acc = np.mean(y_true == y_pred)

    # Top-K accuracy
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -top_k:]
    top_k_acc = np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

    print(f"Accuracy: {acc:.4f}")
    print(f"Top-{top_k} Accuracy: {top_k_acc:.4f}")

    return acc, top_k_acc

def compute_classification_report(y_true, y_pred, class_names):
    """
    Calculer le rapport de classification.
    
    [args]
        y_true (np.ndarray): Les vraies classes.
        y_pred (np.ndarray): Les predictions.
        class_names (list): Les noms des classes.
        
    returns
        report (str): Le rapport de classification.
    """

    report = classification_report(y_true, y_pred, target_names=class_names)
    return report

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir="src/figure", filename="confusion_matrix.png", show=True):
    """
    Plot et sauvegarder la matrice de confusion.
    
    [args]
        y_true (np.ndarray): Les vraies classes.
        y_pred (np.ndarray): Les predictions.
        class_names (list): Les noms des classes.
        save_dir (str): Dossier où sauvegarder la figure.
        filename (str): Nom du fichier image.
        show (bool): Afficher la figure ou non.
    """
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()

def export_results_csv(results_list, filename="model_comparison.csv"):
    """
    Exporter les résultats en CSV.
    
    [args]
        results_list (list): La liste des résultats.
        filename (str): Le nom du fichier CSV, par defaut "model_comparison.csv".
    """
    df = pd.DataFrame(results_list)
    df.to_csv(filename, index=False)
    print(f"CSV exported at {filename}")
    
    
#    ```python
#    # Tableau récapitulatif (100 classes):
#    | Architecture | Top-1 Acc | Top-5 Acc | F1 | Speed (ms) |
#    |---|---|---|---|---|
#    | CNN Simple | 42% | 68% | 0.40 | 2ms |
#    | CNN Profond | 48% | 74% | 0.46 | 5ms |
#    | MobileNetV2 | 52% | 78% | 0.50 | 12ms |
#    | EfficientNetB0 | 58% | 83% | 0.56 | 30ms |
#    | Hiérarchique (bonus) | 55% | 80% | 0.53 | 8ms |
#    ```

def evaluate_and_store(model, model_name, test_ds):
    results = []
    
    y_true, y_pred, y_pred_probs = compute_metrics(model, test_ds)

    acc, top_k_acc = compute_accuracy(y_true, y_pred, y_pred_probs)

    f1 = f1_score(y_true, y_pred, average="weighted")

    # vitesse
    images = [x for x, _ in test_ds.unbatch().take(100)]
    images = tf.stack(images)

    start = time.time()
    model.predict(images, batch_size=32)
    end = time.time()

    speed = (end - start) / 100 * 1000  # ms/image

    results.append({
        "Architecture": model_name,
        "Top-1 Acc": round(acc, 3),
        "Top-5 Acc": round(top_k_acc, 3),
        "F1": round(f1, 3),
        "Speed (ms)": round(speed, 2)
    })
    

if __name__ == "__main__":
    pass
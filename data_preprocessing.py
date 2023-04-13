import os
import numpy as np
import librosa

# Définir les classes de sons à reconnaître
classes = ['chouette_hulotte', 'hunter_shot', 'rouge_gorge']

# Définir les paramètres d'analyse des fichiers audio
sampling_rate = 22050
duration = 5
hop_length = 512
n_mels = 128
n_fft = 2048
n_mfcc = 20

def extract_features(file_path):
    # Charger le fichier audio et fixer sa longueur
    signal, sr = librosa.load(file_path, sr=sampling_rate, duration=duration)
    signal = librosa.util.fix_length(signal, duration * sampling_rate)

    # Extraire les caractéristiques MFCC avec une longueur fixe
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_fft=n_fft,
                                 hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)

    # Ajouter une dimension pour obtenir un tableau de 4 dimensions
    mfccs = np.expand_dims(mfccs, axis=-1)

    return mfccs

def prepare_dataset(data_path):
    X = []
    y = []

    # Parcourir les fichiers audio pour chaque classe
    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_path, cls)
        for file_name in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file_name)

            # Extraire les caractéristiques du fichier audio
            features = extract_features(file_path)

            # Ajouter les caractéristiques et la classe correspondante aux listes
            X.append(features)
            y.append(i)

    # Convertir les listes en tableaux numpy
    X = np.array(X)
    y = np.array(y)

    return X, y

if __name__ == '__main__':
    # Prétraiter les données et les enregistrer dans des fichiers numpy
    X, y = prepare_dataset('data')
    np.save('data_preprocessed/features.npy', X)
    np.save('data_preprocessed/labels.npy', y)
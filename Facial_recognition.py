import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Directorio de las imágenes de entrenamiento
train_data_folder = "clientes"

# Listas para almacenar las características faciales y las etiquetas
X = []
y = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Transferencia de aprendizaje

# Se crea instancia del detector de rostros
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=40,
    thresholds=[0.6, 0.7, 0.7], factor=0.98, post_process=True,
    device=device
)

# Función para extraer el vector de 512 elementos que refleja las características faciales de cada cliente
def extract_features(frame):
    x_aligned, prob = mtcnn(frame, return_prob=True)
    x_embed = None
    if x_aligned is not None:
        print('Rostro detectado con probabilidad: {:8f}'.format(prob))
        x_aligned = torch.stack([x_aligned]).to(device)
        x_embed = model(x_aligned).detach().cpu()
        x_embed = x_embed.numpy()
        return x_embed.ravel()
    else:
        return None

# Recorrer las carpetas de clientes
for client_folder in os.listdir(train_data_folder):
    client_folder_path = os.path.join(train_data_folder, client_folder)
    if os.path.isdir(client_folder_path):
        for filename in os.listdir(client_folder_path):
            if filename.endswith(".png"):
                image_path = os.path.join(client_folder_path, filename)
                frame = cv2.imread(image_path)

                try:
                    x_embed = extract_features(frame)
                    label = client_folder
                    print(label)
                    if x_embed is not None:
                        X.append(x_embed)
                        y.append(client_folder)
                except TypeError as e:
                    print(f"Error al procesar {filename}: {e}")

# Convertir las listas a matrices numpy
X = np.array(X)
y = np.array(y)

# Crear un codificador de etiquetas
label_encoder = LabelEncoder()

# Codificar las etiquetas
y_encoded = label_encoder.fit_transform(y)

# Guardar el LabelEncoder en un archivo .pkl
label_encoder_filename = 'label_encoder.pkl'
with open(label_encoder_filename, 'wb') as file:
    pickle.dump(label_encoder, file)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Entrenar un clasificador RandomForest
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = rf_classifier.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del clasificador RandomForest: {accuracy * 100:.2f}%')

# Visualizar la reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Guardar el modelo en un archivo .pkl
model_filename_pkl = 'rf_model.pkl'
with open(model_filename_pkl, 'wb') as file:
    pickle.dump(rf_classifier, file)

print(f'Modelo RandomForest guardado como {model_filename_pkl}')

# Graficar las características faciales en un espacio bidimensional
plt.figure(figsize=(10, 8))
for label in np.unique(y):
    mask = (y == label)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7)

plt.title('Reducción de dimensionalidad con PCA')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()
plt.show()

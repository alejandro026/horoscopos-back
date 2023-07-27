from flask import Flask, jsonify, request
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins='*')  # Esto habilitará CORS para todas las rutas de tu aplicación Flask

#Datos de prueba
# Datos de ejemplo en forma de lista de listas
datos = [
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
]


@app.route('/kmeans', methods=['GET'])
def kmeans_endpoint():
    
    # Convertir los datos en una matriz NumPy
    matriz_datos = np.array(datos)

    # Configurar el número de clusters k
    k = 4

    # Crear y entrenar el modelo KMeans
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(matriz_datos)

    # Obtener las etiquetas de los clusters
    etiquetas = kmeans.labels_

    # Obtener las coordenadas de los centroides
    centroides = kmeans.cluster_centers_

    #Llamando a la funcion para graficar
    base64_encoded_original = graficaNormal(matriz_datos, kmeans, centroides, k)

    #Llamando a la funcion para graficar con incecia
    base64_encoded_inercia = graficaInercia(matriz_datos)

    #
    base64_encoded_silueta = graficaSilueta(matriz_datos)

    #
    base64_encoded_puntos = graficaPuntos()

    #
    base64_encoded_PCA = graficaPca(matriz_datos, kmeans, etiquetas, k)


    # Devolver los resultados y la gráfica en formato base64
    resultados = {
        'centroidesOriginales': base64_encoded_original,
        'base64_encoded_inercia': base64_encoded_inercia,
        'base64_encoded_silueta': base64_encoded_silueta,
        'base64_encoded_puntos' : base64_encoded_puntos,
        'base64_encoded_PCA': base64_encoded_PCA
        
    }
    return jsonify(resultados)

def graficaNormal(matriz_datos, kmeans, centroides, k):
    #Grafica
    # Graficar los centroides en el espacio original
    plt.figure(figsize=(8, 6))
    for i in range(k):
        plt.scatter(matriz_datos[kmeans.labels_ == i][:, 0], matriz_datos[kmeans.labels_ == i][:, 1], label=f"Cluster {i+1}")
    plt.scatter(centroides[:, 0], centroides[:, 1], s=200, c='black', marker='X', label='Centroides')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Visualización de centroides en el espacio original')
    plt.legend()
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded_original = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_encoded_original

def graficaInercia(matriz_datos):
     # Lista para almacenar las inercias
    inercias = []

    # Rango de valores de k que queremos probar
    valores_k = range(1, 11)

    # Calcular la inercia para cada valor de k
    for k in valores_k:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(matriz_datos)
        inercias.append(kmeans.inertia_)

    # Graficar la inercia para diferentes valores de k
    plt.figure(figsize=(8, 6))
    plt.plot(valores_k, inercias, marker='o')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Inercia para diferentes valores de k')
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded_inercia = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_encoded_inercia

def graficaSilueta(matriz_datos):
    # Lista para almacenar el coeficiente de silueta
    silueta_vals = []

    # Rango de valores de k que queremos probar
    valores_k = range(2, 11)

    # Calcular el coeficiente de silueta para cada valor de k
    for k in valores_k:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(matriz_datos)
        etiquetas = kmeans.labels_
        coeficiente_silueta = silhouette_score(matriz_datos, etiquetas)
        silueta_vals.append(coeficiente_silueta)

    # Graficar el coeficiente de silueta para diferentes valores de k
    plt.figure(figsize=(8, 6))
    plt.plot(valores_k, silueta_vals, marker='o')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Coeficiente de Silueta')
    plt.title('Coeficiente de Silueta para diferentes valores de k')
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded_silueta = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_encoded_silueta

def graficaPuntos():
    # Convertir los datos en una matriz NumPy
    matriz_datos = np.array(datos)

    # Configurar el número de clusters k
    k = 4

    # Crear y entrenar el modelo KMeans
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(matriz_datos)

    # Obtener las etiquetas de los clusters
    etiquetas = kmeans.labels_

    # Contar el número de puntos en cada cluster
    conteo_puntos = np.bincount(etiquetas)

    # Generar la gráfica de barras
    plt.figure(figsize=(8, 6))
    plt.bar(range(k), conteo_puntos, tick_label=[f"Cluster {i+1}" for i in range(k)])
    plt.xlabel('Clusters')
    plt.ylabel('Número de puntos')
    plt.title('Número de puntos en cada cluster')
    plt.grid(axis='y')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded_puntos = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_encoded_puntos

def graficaPca(matriz_datos, kmeans, etiquetas, k):
    # Reducción de dimensionalidad usando PCA
    pca = PCA(n_components=2)
    datos_reducidos = pca.fit_transform(matriz_datos)

    # Obtener las coordenadas de los centroides
    centroides = pca.transform(kmeans.cluster_centers_)

    # Generar la gráfica de dispersión
    plt.figure(figsize=(8, 6))

    # Plotear los puntos de datos según el cluster al que pertenecen
    for i in range(k):
        cluster = datos_reducidos[etiquetas == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i+1}")

    # Plotear los centroides con el mismo color que los puntos de cada cluster
    for i in range(k):
        plt.scatter(centroides[i, 0], centroides[i, 1], s=200, c=f'C{i}', marker='X', label=f'Centroide {i+1}')

    plt.xlabel('Dimensión 1 (PCA)')
    plt.ylabel('Dimensión 2 (PCA)')
    plt.title('Resultados del algoritmo k-means con PCA')
    plt.legend()
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded_PCA = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_encoded_PCA

if __name__ == '__main__':
    app.run(debug=False)

from flask import Flask, jsonify, request
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import base64
from io import BytesIO
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins='*')  # Esto habilitará CORS para todas las rutas de tu aplicación Flask

#Datos de prueba
# Datos de ejemplo en forma de lista de listas
datos = []


@app.route('/kmeans/<int:clusterNum>', methods=['GET'])
def kmeans_endpoint(clusterNum):
    
    # Convertir los datos en una matriz NumPy
    matriz_datos = np.array(datos)

    # Configurar el número de clusters k
    k = clusterNum

    # Crear y entrenar el modelo KMeans
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(matriz_datos)

    # Obtener las etiquetas de los clusters
    etiquetas = kmeans.labels_

    # Obtener las coordenadas de los centroides
    centroides = kmeans.cluster_centers_

    # Llamando a la funcion para graficar
    base64_encoded_original = graficaNormal(matriz_datos, kmeans, centroides, k)

    #Llamando a la funcion para graficar con incecia
    base64_encoded_inercia = graficaInercia(matriz_datos)

    #
    base64_encoded_silueta = graficaSilueta(matriz_datos)

    #
    base64_encoded_puntos = graficaPuntos(kmeans, k)

    #
    base64_encoded_PCA = graficaPca(matriz_datos, kmeans, etiquetas, k)

    #
    base64_encoded_pastel = graficaPastel(etiquetas, k)


    # Devolver los resultados y la gráfica en formato base64
    resultados = {
        'centroidesOriginales': base64_encoded_original,
        'base64_encoded_inercia': base64_encoded_inercia,
        'base64_encoded_silueta': base64_encoded_silueta,
        'base64_encoded_puntos' : base64_encoded_puntos,
        'base64_encoded_PCA': base64_encoded_PCA,
        'base64_encoded_pastel': base64_encoded_pastel
        
    }
    return jsonify(resultados)

@app.route('/recibir_json', methods=['POST'])
def recibir_json():
    dataR = request.json
    arreglo_datos = [registro for registro in dataR['datosFinales']]
    #Signos seleccionados
    global signosSeleccionados  
    signosSeleccionados= dataR['signosSeleccionados']
    global datos
    datos = [convertir_objeto(objeto) for objeto in arreglo_datos]
    return jsonify({"mensaje": "JSON recibido exitosamente", "arreglo_datos": datos})

# Función para convertir el objeto en el formato deseado
def convertir_objeto(objeto):
    preguntas = []
    for i in range(2, 13):
        pregunta_key = f"pregunta{i}"
        if pregunta_key in objeto:
            preguntas.append(objeto[pregunta_key])
    return preguntas


# Función para aplicar el algoritmo KMeans a los datos filtrados
def aplicar_kmeans(datos_filtrados):
    k = 6
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(datos_filtrados)
    etiquetas = kmeans.labels_
    centroides = kmeans.cluster_centers_
    return {'kmeans': kmeans, 'etiquetas': etiquetas, 'centroides': centroides, 'k': k}


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

def graficaPuntos(kmeans, cluster):
    # Convertir los datos en una matriz NumPy
    matriz_datos = np.array(datos)

    # Configurar el número de clusters k
    k = cluster

    # # Crear y entrenar el modelo KMeans
    # kmeans = KMeans(n_clusters=k)
    # kmeans.fit(matriz_datos)

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

def graficaPastel(etiquetas, k):
    # Asignar cada dato a su respectivo grupo

    # etiquetas_personalizadas = [
    #     "Aries", "Tauro", "Géminis", "Cáncer", "Leo", "Virgo"
    # ]
    etiquetas_personalizadas= signosSeleccionados

    # Asignar cada dato a su respectivo grupo con etiquetas personalizadas
    datos_con_grupo = []
    for i, dato in enumerate(datos):
        grupo = etiquetas[i]
        signo_zodiaco = etiquetas_personalizadas[grupo]
        datos_con_grupo.append((dato, grupo, signo_zodiaco))

    # Contar la cantidad de datos en cada grupo
    conteo_por_grupo = [0] * k
    for dato, grupo, _ in datos_con_grupo:
        conteo_por_grupo[grupo] += 1

    # Calcular el porcentaje de datos en cada grupo
    porcentaje_por_grupo = [conteo * 100 / len(datos) for conteo in conteo_por_grupo]

    # Configurar el tamaño de la figura
    plt.figure(figsize=(8, 8))

    # Generar la gráfica de pastel
    plt.pie(porcentaje_por_grupo, labels=[f'{etiquetas_personalizadas[i]}' for i in range(k)], autopct='%1.1f%%', startangle=140)

    # Agregar título a la gráfica con un salto de línea
    plt.title('Distribución de Datos en los Grupos\n\n')

    # Mostrar la gráfica
    plt.axis('equal')  # Esto garantiza que el gráfico sea un círculo en lugar de una elipse
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded_pastel = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_encoded_pastel

def grafica_pastel_agg(etiquetas, k):
    # Contar la cantidad de datos en cada grupo
    conteo_por_grupo = np.bincount(etiquetas)

    # Calcular el porcentaje de datos en cada grupo
    porcentaje_por_grupo = conteo_por_grupo * 100 / len(datos)

    # Configurar el tamaño de la figura
    plt.figure(figsize=(8, 8))

    # Generar la gráfica de pastel
    plt.pie(porcentaje_por_grupo, labels=[f'Grupo {i}' for i in range(k)], autopct='%1.1f%%', startangle=140)

    # Agregar título a la gráfica con un salto de línea
    plt.title('Distribución de Datos en los Grupos (Agglomerative Clustering)\n\n')

    # Mostrar la gráfica
    plt.axis('equal')  # Esto garantiza que el gráfico sea un círculo en lugar de una elipse
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded_pastel_agg = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_encoded_pastel_agg

def grafica_pca(matriz_datos, etiquetas, k):
    # Reducir la dimensionalidad de los datos a 2D mediante PCA
    pca = PCA(n_components=2)
    datos_pca = pca.fit_transform(matriz_datos)

    # Crear un diccionario para almacenar los datos de cada cluster por separado
    datos_por_cluster = {i: [] for i in range(k)}
    for i, dato in enumerate(datos_pca):
        grupo = etiquetas[i]
        datos_por_cluster[grupo].append(dato)

    # Configurar el tamaño de la figura
    plt.figure(figsize=(8, 8))

    # Generar el gráfico de dispersión para cada cluster
    for i in range(k):
        datos_cluster = np.array(datos_por_cluster[i])
        plt.scatter(datos_cluster[:, 0], datos_cluster[:, 1], label=f'Cluster {i+1}')

    # Agregar etiquetas para los clusters en la leyenda
    plt.legend()

    # Agregar título al gráfico
    plt.title('Gráfico PCA de los Clusters (Agglomerative Clustering)\n\n')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded_pca_agg = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return base64_encoded_pca_agg

@app.route('/agglomerativeClustering/<int:clusterNum>', methods=['GET'])
def agglomerativeClustering(clusterNum):
    # Convertir los datos en una matriz NumPy
    matriz_datos = np.array(datos)

    # Configurar el número de clusters k (ajustar según necesites)
    k = clusterNum

    # Crear y entrenar el modelo Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=k)
    etiquetas = agglomerative.fit_predict(matriz_datos)
    
    #Llamando a la funcion para graficar con incecia
    # base64_encoded_inercia = graficaInercia(matriz_datos)

    #
    # base64_encoded_silueta = graficaSilueta(matriz_datos)

    #
    # base64_encoded_puntos = graficaPuntos(agglomerative, k)

    #
    base64_encoded_PCA = grafica_pca(matriz_datos, etiquetas, k)

    #
    base64_encoded_pastel = grafica_pastel_agg(etiquetas, k)


    # Devolver los resultados y la gráfica en formato base64
    resultados = {
        # 'base64_encoded_inercia': base64_encoded_inercia,
        # 'base64_encoded_silueta': base64_encoded_silueta,
        # 'base64_encoded_puntos' : base64_encoded_puntos,
        'base64_encoded_PCA': base64_encoded_PCA,
        'base64_encoded_pastel': base64_encoded_pastel
        
    }
    return jsonify(resultados)



if __name__ == '__main__':
    app.run(debug=True)

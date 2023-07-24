from flask import Flask, jsonify, request
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins='*')  # Esto habilitará CORS para todas las rutas de tu aplicación Flask

# Datos de ejemplo
horoscopos = ['Aries', 'Tauro', 'Géminis', 'Cáncer', 'Leo', 'Virgo', 'Libra', 'Escorpio', 'Sagitario', 'Capricornio', 'Acuario', 'Piscis']
caracteristicas = np.array([[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],    # Aries
                            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],    # Tauro
                            [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0],    # Géminis
                            [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],    # Cáncer
                            [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],    # Leo
                            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],    # Virgo
                            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],    # Libra
                            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],    # Escorpio
                            [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],    # Sagitario
                            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],    # Capricornio
                            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],    # Acuario
                            [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1]])   # Piscis

# Configuración del algoritmo k-means
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(caracteristicas)

@app.route('/kmeans', methods=['POST'])
def kmeans_endpoint():
    # Obtener los datos enviados desde Angular
    #data = request.get_json()
    data = horoscopos
    nuevas_caracteristicas = caracteristicas

    # Realizar el cálculo del K-means con los nuevos datos
    nuevas_etiquetas = kmeans.predict(nuevas_caracteristicas)

    # Obtener los horóscopos del cluster
    clusters = {}
    for i in range(k):
        cluster = np.where(nuevas_etiquetas == i)[0]
        horoscopos_cluster = [horoscopos[j] for j in cluster]
        clusters[f'Cluster {i+1}'] = horoscopos_cluster

    # Generar una gráfica (ejemplo con Matplotlib)
    plt.figure()
    plt.scatter(nuevas_caracteristicas[:, 0], nuevas_caracteristicas[:, 1], c=nuevas_etiquetas)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Devolver los resultados y la gráfica en formato base64
    resultados = {
        'clusters': clusters,
        'grafica': base64_encoded
    }
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)

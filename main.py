from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import csv

app = Flask(__name__)

# Charger le modèle
model = hub.load("https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1")

# Charger et prétraiter l'image
def preprocess_image(image):
    image = image.resize((224, 224))  # Redimensionner l'image
    image = np.array(image) / 255.0  # Normaliser les valeurs de pixel entre 0 et 1
    return image

# Prédiction de l'image
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    expanded_dims_image = np.expand_dims(preprocessed_image, axis=0)  # Ajouter une dimension supplémentaire pour l'entrée du modèle
    predictions = model.signatures["default"](tf.constant(expanded_dims_image, dtype=tf.float32))
    predicted_class = np.argmax(predictions["default"], axis=-1)[0]  # Trouver la classe prédite avec la plus haute probabilité
    probability = np.max(predictions["default"])  # Obtenir la probabilité de la classe prédite
    return predicted_class, probability

# Obtenir le nom de la classe à partir de l'ID
def get_class_name(class_id):
    with open('label.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Ignorer la première ligne (entête)
        for row in reader:
            if int(row[0]) == class_id:
                return row[1]
    return "Classe non trouvée"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image = Image.open(request.files['image'])
    predicted_class, probability = predict_image(image)
    class_name = get_class_name(predicted_class)

    response = {
        'class_name': class_name,
        'probability': float(probability)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()

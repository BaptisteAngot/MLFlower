# Projet de Classification d'Images de Plantes
Ce projet est une démonstration de classification d'images de plantes en utilisant TensorFlow et TensorFlow Hub. Il utilise un modèle pré-entrainé disponible sur TensorFlow Hub pour prédire la classe d'une image de plante donnée.

## Installation
1. Assurez-vous d'avoir installé Python (version 3.6 ou supérieure) sur votre système.
2. Clonez ou téléchargez ce dépôt de projet.
```bash
git clone https://github.com/BaptisteAngot/MLFlower
```

3. Accédez au répertoire du projet.
```bash
cd MLFlower
```
4. Installez les dépendances requises à l'aide de pip, en utilisant le fichier requirements.txt.
```bash
pip install -r requirements.txt
```

## Utilisation de l'API
L[.gitignore](.gitignore)'API fournit une route POST /predict qui permet de soumettre une image pour classification. Suivez les étapes ci-dessous pour utiliser l'API :

1. Assurez-vous que le serveur de l'API est en cours d'exécution en exécutant le fichier api.py.
```
python main.py
```
2. Envoyez une requête POST à l'URL http://localhost:5000/predict en incluant une image à classer. Vous pouvez utiliser des outils tels que cURL, Postman ou envoyer la requête depuis votre propre application. Assurez-vous d'inclure l'image en tant que fichier dans le corps de la requête.
Exemple d'utilisation de cURL : 
```bash
curl -X POST -F "image=@/chemin/vers/votre/image.jpg" http://localhost:5000/predict
```

3. L'API renverra une réponse au format JSON contenant la classe prédite et le pourcentage de précision de la prédiction.
Exemple de réponse JSON :
```json
{
  "class_name": "Rose",
  "probability": 0.987
}
```
## Remarque
Assurez-vous d'avoir des images de plantes dans un format pris en charge (par exemple, JPEG, PNG) pour obtenir des prédictions précises. Vérifiez également que les images sont suffisamment claires et qu'elles contiennent principalement des plantes pour de meilleurs résultats de classification.

N'hésitez pas à explorer et à modifier le code pour répondre à vos besoins spécifiques. Amusez-vous bien avec la classification d'images de plantes !
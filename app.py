from flask import Flask, request, jsonify
import joblib

# 1. créer une instance de la classe Flask
app = Flask(__name__)

# 2. Definition de la fonction predictive
    # fonction Python qui accepte en entrée notre modèle d'apprentissage automatique entraîné 
    # et certaines données d'entrée puis renvoi la prédiction du modèle.
def return_prediction(model, input_json):

    input_data = [[input_json[k] for k in input_json.keys()]]
    prediction = model.predict(input_data)[0]

    return prediction

# 3. chargez notre modèle de prédicteur de l'âge des ormeaux
model = joblib.load('abalone_predictor.joblib')

# 4. configurer notre page d'accueil
@app.route("/")
def index():
    return """
    <h1>Bienvenue dans notre service de prédiction d'ormeaux</h1>
    Pour utiliser ce service, effectuez une demande de publication JSON à l'URL /predict avec les champs suivants :
    <ul>
    <li>longueur</li>
    <li>diamètre</li>
    <li>hauteur</li>
    <li>poids_entier</li>
    </ul>
    """

# 5. définir une nouvelle route qui acceptera les requêtes POST et renverra nos prédictions de modèle
@app.route('/predict', methods=['POST'])
def abalone_prediction():
    content = request.json
    results = return_prediction(model, content)
    return jsonify(results)

# 6. nous permet d'exécuter flask en utilisant python app.py
if __name__ == '__main__':
    app.run()

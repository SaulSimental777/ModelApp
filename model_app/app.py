from flask import Flask, render_template, request, current_app, g, jsonify
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from modelClass import RetrievalModel
from dotenv import load_dotenv
import os
import json

from flask_pymongo import pymongo



load_dotenv()
app = Flask(__name__)
port = int(os.getenv("PORT", 3100))

#-- Conexion a MongoDB --#
CONNECTION_STRING = os.environ.get("MONGO_URL")

client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('test')


#--- Solicitud de los datos de interaccion y comics --#
interactions = [i for i in db.interactions.find({})]
comics = [j for j in db.products.find({})]

df_comics = []
df_interactions = []

for k in interactions:
    df_interactions.append((k['user_id'], k['product_id'], k['score']))

for j in comics:
    df_comics.append((j['productModel_id'],j['title']))


#--- Preprocesamiento de los datos ---#
def preprocess_rating(row):
    user_id, comic_id, score = row
    return(
        tf.strings.to_number(str(user_id), out_type= tf.int32),
        {
            "comic_id": tf.strings.to_number(str(comic_id), out_type=tf.int32),
            "score": tf.cast(score, dtype= tf.float32)
        }
    )

processed_interactions = [preprocess_rating(row) for row in df_interactions]

#--- Aplicacion de reorganizacion de datos ---# 
shuffled_interactions = tf.data.Dataset.from_generator(
    lambda: processed_interactions,
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.int32),
        {
            "comic_id": tf.TensorSpec(shape=(), dtype=tf.int32),
            "score": tf.TensorSpec(shape=(), dtype=tf.float32)
        }
    )
).shuffle(100_000, seed=42, reshuffle_each_iteration= False)


#--- Separacion de los datos de entrenamiento y validacion ---#
train_interactions = shuffled_interactions.take(int(0.8 * len(processed_interactions)))
test_interactions = shuffled_interactions.skip(int(0.8 * len(processed_interactions)))

train_interactions = train_interactions.batch(100).cache()
test_interactions = test_interactions.batch(100).cache()


#--- Obtencion del titulo del comic en base a su id ---#
comic_id_to_comic_title = {int(str(row[0])): row[1] for row in df_comics}
comic_id_to_comic_title[0] = ""


#--- Obtencion de los id unicos para usuarios y comics ---#
user_ids = {row[0] for row in df_interactions}
comic_ids = {row[1] for row in df_interactions}

users_count = len(user_ids)
comics_count = len(comic_ids)

#--- Modelo ---#
model = RetrievalModel(users_count + 1, comics_count + 1)

#--- Prediccion ---#
def ValuePrediction(user_id):

    predictions = model.predict(keras.ops.convert_to_tensor([user_id]))
    predictions = keras.ops.convert_to_numpy(predictions["predictions"])
    
    recommended_comics = []
    for comic_id in predictions[0]:
        recommended_comics.append(comic_id_to_comic_title.get(int(comic_id),"Desconocido"))

    return recommended_comics

@app.route("/recommend", methods =["POST"])
def recommend():
    try:
        data = request.get_json()
        user_id = int(data.get("user_id"))

        print("IDENTIFICADOR RECIBIDO: ", user_id)

        recommendations = ValuePrediction(user_id)

        return jsonify({
            "status": "success",
            "user_id":user_id,
            "recommendations": recommendations
        })
    
    except Exception as e:
        return jsonify({"status":"error", "message": str(e)})

    

if __name__ == "__main__":
    app.run(port= port, debug=True)
    

    
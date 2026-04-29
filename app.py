## Example of a post request: "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"

from flask import Flask, render_template, request, jsonify
import numpy as np  
from tensorflow.keras.models import load_model
import joblib

def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    
    flower = [[s_len,s_wid,p_len,p_wid]]
    
    flower = scaler.transform(flower)
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    class_ind = model.predict(flower)

    class_ind = np.round(class_ind)

    class_ind = np.argmax(class_ind, axis=1)
    print(f"prediction is {class_ind}")
    
    return classes[class_ind][0].item()

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
flower_model = load_model("final_iris_model.keras")
flower_scaler = joblib.load("iris_scaler.pkl")

@app.route('/api/flower', methods=['POST'])
def predict_flower():

    content = request.json
    
    results = return_prediction(model=flower_model,scaler=flower_scaler,sample_json=content)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
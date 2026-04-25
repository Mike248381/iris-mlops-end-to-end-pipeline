import argparse
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def return_prediction(model, scaler, sample_json):
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    
    flower = [[s_len, s_wid, p_len, p_wid]]
    flower = scaler.transform(flower)
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    class_ind = model.predict(flower)
    class_ind = np.round(class_ind)
    class_ind = np.argmax(class_ind, axis=1)
    
    return classes[class_ind][0].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris prediction from JSON input")
    
    parser.add_argument(
        "--input_json",
        type=str,
        help="JSON string with flower features"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to JSON file with flower features"
    )
    
    args, unknown = parser.parse_known_args()
    
    # Ensure at least one input is provided
    if not args.input_json and not args.input_file:
        raise ValueError("Provide either --input_json or --input_file")
    
    # Load JSON from string or file
    if args.input_json:
        sample_json = json.loads(args.input_json)
    else:
        with open(args.input_file, "r") as f:
            sample_json = json.load(f)
    
    # Load model and scaler once
    model = load_model("final_iris_model.keras")
    scaler = joblib.load("iris_scaler.pkl")
    
    prediction = return_prediction(model, scaler, sample_json)
    
    print(f"Prediction: {prediction}")
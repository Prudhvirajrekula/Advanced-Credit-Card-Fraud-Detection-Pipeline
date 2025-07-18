#!/usr/bin/env python
import os
import joblib
import numpy as np
import json
from flask import Flask, request, jsonify
from werkzeug.serving import run_simple

# Load the model at container startup.
model_path = os.path.join(os.environ.get('MODEL_DIR', '/opt/ml/model'), 'model.joblib')
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint."""
    return '', 200

@app.route('/invocations', methods=['POST'])
def predict():
    """Prediction endpoint."""
    if request.content_type == 'application/json':
        data = request.get_json()
        # Expecting a JSON payload with a key "instances" that is a list of records.
        input_data = data.get('instances')
        if input_data is None:
            return jsonify({'error': 'No instances provided'}), 400

        # Convert list of dicts to a NumPy array
        try:
            input_array = np.array([list(record.values()) for record in input_data])
        except Exception as e:
            return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    else:
        return jsonify({'error': 'Unsupported content type: ' + request.content_type}), 415

    predictions = model.predict(input_array)
    probabilities = model.predict_proba(input_array)[:, 1].tolist()

    return jsonify({'predictions': predictions.tolist(), 'probabilities': probabilities})

if __name__ == '__main__':
    # Use run_simple for local testing; in SageMaker, the container will call app.run().
    run_simple('0.0.0.0', 8080, app)


from flask import Flask, request, jsonify
import pickle
import json
import os
import joblib
import uuid
import pandas as pd
from peewee import SqliteDatabase, Model, IntegerField, FloatField, TextField, BooleanField, IntegrityError
from utils import FeatureCreationAPI, generate_unique_id
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

# Define database model
#DB = SqliteDatabase('predictions.db')
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predicts.db')

##3###################################################
class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    pred = BooleanField()
    pred_proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# Initialize Flask application
app = Flask(__name__)

# Load the column names from the file
with open(os.path.join('data', 'columns.json'), 'r') as fh:
    columns = json.load(fh)

# Load the dtypes from the file
with open(os.path.join('data', 'dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)

# Load the pipeline from the file (corrected single load)
with open(os.path.join('data', 'pipeline.pickle'), 'rb') as fh:
    pipeline = pickle.load(fh)


@app.route('/will_recidivate/', methods=['POST'])
def will_recidivate():
    app.logger.info(f"Received request for endpoint: {request.endpoint}")

    try:
        # Extract data from request
        data_json = request.get_json()
        app.logger.info(f"Data received: {data_json}")
        
        
        # Generate unique ID if not provided
        if not data_json.get('id'):
            data_json['id'] = generate_unique_id()
            app.logger.info(f"Generated unique ID for missing value ID: {data_json['id']}")
            #app.logger.info(f"Data received after new id was generated: {data_json}")
            
        # Convert data to DataFrame
        data_df = pd.DataFrame([data_json])
        
        if data_df.isnull().values.any():
            app.logger.error(f"Input data contains null values: {data_df.isnull().sum()}")
            # Handle missing values: fill with default values
            data_df.fillna({
                'c_jail_in': '2013-09-13 02:36:35.500'
            }, inplace=True)
            app.logger.info(f"Data after filling missing values: {data_df['c_jail_in']}")

        # Apply feature creation pipeline
        processed_data = FeatureCreationAPI().transform(data_df)

        # Extract the ID from the DataFrame
        _id = processed_data['id'].values[0]
        app.logger.info(f"Observation ID: {_id}")
        observation = processed_data[['id', 'sex', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                                      'priors_count', 'c_charge_degree', 'age_group', 'weights_race']]

        # Prepare observation as DataFrame with correct columns and types
        obs = pd.DataFrame(observation.values,
                           columns=observation.columns).astype(dtypes)

        # Predict the class
        pred = pipeline.predict(obs)[0]
        #app.logger.info(f"Predicted class: {pred}")

        # Predict probability of positive class (index 1)
        pred_proba = pipeline.predict_proba(obs)[0, 1]
        #app.logger.info(f"Predicted probability: {pred_proba}")

        # Save prediction to database
        p = Prediction(
            observation_id=int(_id),
            pred=bool(pred),
            pred_proba=float(pred_proba),
            observation=observation.to_json(orient='records')[1:-1]
        )

        try:
            p.save()
        except IntegrityError:
            error_msg = f'Observation ID: {_id} already exists'
            app.logger.error(f'ERROR: {error_msg}')
            DB.rollback()
            return jsonify({'error': error_msg, 'outcome': bool(pred)}), 409

        response_data = {
            'id': int(_id),
            'outcome': bool(pred)
        }
        
        # Log the JSON response
        app.logger.info(f"Response JSON: {response_data}")
        
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({"error": "Observation is invalid!"})


@app.route('/recidivism_result/', methods=['POST'])
def recidivism_result():
    app.logger.info(f"Received request for endpoint: {request.endpoint}")

    try:
        # Extract data from request
        data_json = request.get_json()
        app.logger.info(f"Data received: {data_json}")
        
        # Check if 'id' is None or not present
        if 'id' not in data_json or data_json['id'] is None:
            error_msg = "'id' field is missing or set to None"
            app.logger.error(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg})

        # Extract ID and true outcome from the request
        observation_id = int(data_json['id'])
        true_outcome = bool(data_json['outcome'])

        # Check if the ID exists in the database
        prediction = Prediction.get_or_none(
            Prediction.observation_id == observation_id)

        if prediction is None:
            error_msg = f'Observation ID: {observation_id} not found'
            app.logger.error(f'ERROR: {error_msg}')
            return jsonify({'error': error_msg})

        # Update the true_class field in the database
        prediction.true_class = true_outcome
        prediction.save()

        response_data = {
            'id': prediction.observation_id,
            'outcome': prediction.true_class,
            'predicted_outcome': prediction.pred
        }

        # Log the JSON response
        app.logger.info(f"Response JSON: {response_data}")
        
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({"error": "Invalid input data!"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)

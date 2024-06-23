import os
import pickle
import json
import logging
import pandas as pd
from flask import Flask, Blueprint, request, jsonify
from peewee import SqliteDatabase, Model, IntegerField, FloatField, TextField, BooleanField, IntegrityError
from playhouse.db_url import connect
from utils import FeatureCreationAPI, generate_unique_id

# Database configuration
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predicts.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    pred = BooleanField()
    pred_proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# Initialize Flask application and Blueprint
app = Flask(__name__)
routes = Blueprint('routes', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load column names and dtypes
with open(os.path.join('data', 'columns.json'), 'r') as fh:
    columns = json.load(fh)

with open(os.path.join('data', 'dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)

with open(os.path.join('data', 'pipeline.pickle'), 'rb') as fh:
    pipeline = pickle.load(fh)

def load_json(request):
    try:
        return request.get_json()
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return None

def validate_input(data_json, required_fields):
    for field in required_fields:
        if field not in data_json or data_json[field] is None:
            logger.error(f"Invalid input: '{field}' field is missing or set to None")
            return False, f"'{field}' field is missing or set to None"
    return True, ""

def handle_missing_data(data_df):
    if data_df.isnull().values.any():
        data_df.fillna({'c_jail_in': '2013-09-13 02:36:35.500'}, inplace=True)
    return data_df

def process_data(data_json):
    logger.info("Processing data...")
    data_df = pd.DataFrame([data_json])
    data_df = handle_missing_data(data_df)
    processed_data = FeatureCreationAPI().transform(data_df)
    return processed_data

def save_prediction(_id, pred, pred_proba, observation):
    p = Prediction(
        observation_id=int(_id),
        pred=bool(pred),
        pred_proba=float(pred_proba),
        observation=observation.to_json(orient='records')[1:-1]
    )
    try:
        p.save()
        logger.info(f"Saved prediction for observation ID: {_id}")
    except IntegrityError:
        error_msg = f'Observation ID: {_id} already exists'
        logger.warning(error_msg)
        DB.rollback()
        return error_msg

@routes.route('/recidivism_result/', methods=['POST'])
def recidivism_result():
    logger.info("Received POST request on '/recidivism_result/' endpoint.")
    data_json = load_json(request)
    if data_json is None:
        logger.error("Invalid JSON input received.")
        return jsonify({"error": "Invalid JSON input!"}), 400

    valid, message = validate_input(data_json, ['id', 'outcome'])
    if not valid:
        logger.error(f"Validation failed: {message}")
        return jsonify({'error': message}), 400

    observation_id = int(data_json['id'])
    true_outcome = bool(data_json['outcome'])

    prediction = Prediction.get_or_none(Prediction.observation_id == observation_id)
    if prediction is None:
        logger.warning(f"Observation ID: {observation_id} not found")
        return jsonify({'error': f'Observation ID: {observation_id} not found'}), 404

    prediction.true_class = true_outcome
    prediction.save()

    response_data = {
        'id': prediction.observation_id,
        'outcome': prediction.true_class,
        'predicted_outcome': prediction.pred
    }

    logger.info(f"Response generated for '/recidivism_result/' endpoint for observation ID: {observation_id}")
    return jsonify(response_data)

@routes.route('/will_recidivate/', methods=['POST'])
def will_recidivate():
    logger.info("Received POST request on '/will_recidivate/' endpoint.")
    data_json = load_json(request)
    if data_json is None:
        logger.error("Invalid JSON input received.")
        return jsonify({"error": "Invalid JSON input!"}), 400

    if not data_json.get('id'):
        data_json['id'] = generate_unique_id()

    processed_data = process_data(data_json)
    _id = processed_data['id'].values[0]
    observation = processed_data[['id', 'sex', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                                  'priors_count', 'c_charge_degree', 'age_group', 'weights_race']]
    obs = pd.DataFrame(observation.values, columns=observation.columns).astype(dtypes)

    try:
        pred = pipeline.predict(obs)[0]
        pred_proba = pipeline.predict_proba(obs)[0, 1]
        logger.info(f"Prediction successful for observation ID: {_id}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed!"}), 500

    error_msg = save_prediction(_id, pred, pred_proba, observation)
    if error_msg:
        logger.warning(error_msg)
        return jsonify({'error': error_msg, 'outcome': bool(pred)}), 409

    response_data = {
        'id': int(_id),
        'outcome': bool(pred)
    }

    logger.info(f"Response generated for '/will_recidivate/' endpoint for observation ID: {_id}")
    return jsonify(response_data)

# Register blueprint
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)

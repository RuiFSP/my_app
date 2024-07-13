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
    """
    Load JSON data from a Flask request object.

    Args:
        request (flask.Request): The Flask request object containing JSON data.

    Returns:
        dict: Parsed JSON data as a dictionary, or None if parsing fails.
    """
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("No JSON data found")
        return data
    except ValueError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def validate_input(data_json, required_fields):
    """
    Validate the presence and non-null values of required fields in JSON data.

    Args:
        data_json (dict): JSON data to validate.
        required_fields (list): List of required field names.

    Returns:
        tuple: (bool, str) - A boolean indicating validation success, and an error message if validation fails.
    """
    errors = []
    for field in required_fields:
        if field not in data_json or data_json[field] is None:
            errors.append(f"'{field}' field is missing or set to None")
    if errors:
        logger.error(f"Invalid input: {errors}")
        return False, ", ".join(errors)
    return True, ""

def handle_missing_data(data_df):
    """
    Handle missing data in the provided DataFrame by filling specific fields with default values.

    Args:
        data_df (pd.DataFrame): DataFrame to handle missing data.

    Returns:
        pd.DataFrame: DataFrame with missing data handled.
    """
    if data_df.isnull().values.any():
        data_df.fillna({'c_jail_in': '2013-09-13 02:36:35.500'}, inplace=True)
    return data_df

def process_data(data_json):
    """
    Process raw JSON data into a format suitable for model prediction.

    Args:
        data_json (dict): Raw JSON data to process.

    Returns:
        pd.DataFrame: Processed data ready for prediction.
    """
    logger.info("Processing data...")
    data_df = pd.DataFrame([data_json])
    data_df = handle_missing_data(data_df)
    processed_data = FeatureCreationAPI().transform(data_df)
    return processed_data

def save_prediction(_id, pred, pred_proba, observation):
    """
    Save prediction results to the database.

    Args:
        _id (int): Unique observation ID.
        pred (bool): Prediction result.
        pred_proba (float): Prediction probability.
        observation (pd.DataFrame): Original observation data.

    Returns:
        str: Error message if saving fails, otherwise None.
    """
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
    """
    Handle POST requests for recidivism results.

    Expects a JSON payload with 'id' and 'outcome' fields.

    Returns:
        Response: JSON response containing the prediction outcome or an error message.
    """
    logger.info("Received POST request on '/recidivism_result/' endpoint.")
    data_json = load_json(request)
    logger.info(f"Request JSON: {data_json}")
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
    logger.info(f"Response JSON: {response_data}")
    return jsonify(response_data)

@routes.route('/will_recidivate/', methods=['POST'])
def will_recidivate():
    """
    Handle POST requests to predict recidivism.

    Expects a JSON payload with observation data.

    Returns:
        Response: JSON response containing the prediction result or an error message.
    """
    logger.info("Received POST request on '/will_recidivate/' endpoint.")
    data_json = load_json(request)
    logger.info(f"Request JSON: {data_json}")
    if data_json is None:
        logger.error("Invalid JSON input received.")
        return jsonify({"error": "Invalid JSON input!"}), 400

    if not data_json.get('id'):
        data_json['id'] = generate_unique_id()

    processed_data = process_data(data_json)
    _id = processed_data['id'].values[0]
    observation = processed_data[['id', 'sex', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                                  'priors_count', 'c_charge_degree', 'age_group', 'weights_race']]
    
    try:
        # Ensure only existing columns are casted
        obs = pd.DataFrame(observation.values, columns=observation.columns)
        for column in obs.columns:
            if column in dtypes:
                obs[column] = obs[column].astype(dtypes[column])
        
        pred = pipeline.predict(obs)[0]
        pred_proba = pipeline.predict_proba(obs)[0, 1]
        
        # Add pred_prob and pred_baseline columns to the observation
        obs['pred_prob'] = pred_proba
        obs['pred_baseline'] = pred
        
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
    logger.info(f"Response JSON: {response_data}")
    return jsonify(response_data)

# Register blueprint
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)

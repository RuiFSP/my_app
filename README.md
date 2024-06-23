# Recidivism Prediction API

This repository contains the code for a Flask-based API that predicts the likelihood of recidivism based on provided input data. The API provides endpoints to submit data for prediction and to retrieve prediction results.

## Table of Contents

- Installation
- Configuration
- Usage
- Endpoints
- Database Schema
- Utilities
- Logging
- License

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-repo/recidivism-prediction-api.git
cd recidivism-prediction-api
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages (DEV): 

```bash
pip install -r requirements_dev.txt
```

4. Ensure you have the necessary data files in the data directory:

- `columns.json`
- `dtypes.pickle`
- `pipeline.pickle`

## Configuration

The application relies on environment variables for configuration. The primary environment variable is `DATABASE_URL`, which specifies the database connection string. If not set, the application defaults to using a local SQLite database (`sqlite:///predicts.db`).

## Usage

To start the Flask application, run the following command:

```bash
python app.py
```

The API will be available at `http://0.0.0.0:5000`

## Endpoints

### `/will_recidivate/` [POST]

Predicts the likelihood of recidivism for the given input data.

- Request Body: JSON

```json
{
    "id": 1234,
    "name": "rui antonio",
    "sex": "Male",
    "dob": "1987-06-21",
    "race": "Caucasian",
    "juv_fel_count": 0,
    "juv_misd_count": 3,
    "juv_other_count": 0,
    "priors_count": 5,
    "c_case_number": "13022428MM10A",
    "c_charge_degree": "F",
    "c_charge_desc": "Driving Under The Influence",
    "c_offense_date": "2013-08-19",
    "c_arrest_date": "2013-09-07",
    "c_jail_in": "2013-09-20 06:32:00",
}
```

- Response: JSON

```json
{
  "id": 1234,
  "outcome": true
}
```
### `/recidivism_result/` [POST]

Updates the true outcome for a prediction and retrieves the stored prediction.

```json
{
  "id": 1234,
  "outcome": true
}
```

- Response: JSON

```json
{
  "id": 1234,
  "outcome": true,
  "predicted_outcome": false
}
```

## Database Schema

The application uses a SQLite database (or other databases via the DATABASE_URL environment variable) with the following schema:

### `Prediction`
- `observation_id` (Integer, unique)
- `observation` (Text)
- `pred` (Boolean)
- `pred_proba` (Float)
- `true_class` (Integer, nullable)

## Utilities

### `load_json(request)`

Loads JSON data from a Flask request.

### `validate_input(data_json, required_fields)`

Validates the input JSON to ensure all required fields are present.

### `handle_missing_data(data_df)`

Handles missing data in the input DataFrame by filling in default values.

### `process_data(data_json)`

Processes the input data, transforms it, and prepares it for prediction.

### `save_prediction(_id, pred, pred_proba, observation)`

Saves a prediction to the database.

## Logging

The application uses Python's built-in logging module to log information. Logs are configured to display messages at the INFO level and higher.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
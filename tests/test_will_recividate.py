import requests
import uuid

# Define the data to send in the request
data = {
    "id": str(int(uuid.uuid4().int & (1 << 25)-1)),
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
    "c_offense_date": None,
    "c_arrest_date": "2013-08-19",
    "c_jail_in": "2013-08-20 06:32:00",
}

# URL of your Flask application endpoint
# url = 'https://ruipinto-ldsa-production.up.railway.app/will_recidivate/'
url = 'http://127.0.0.1:5000/will_recidivate/'

# Function to send the POST request and print the result


def test_endpoint(data):
    response = requests.post(url, json=data)
    print(f"Testing with payload: {data}")
    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    print("-" * 50)

# Test different scenarios


# Scenario 1: Valid ID and c_jail_in normal point
test_endpoint(data)

# Scenario 2: ID that does not exist in the database
data["id"] = str(int(uuid.uuid4().int & (1 << 25)-1))
test_endpoint(data)

# Scenario 3: ID is None
data["id"] = None
test_endpoint(data)

# Scenario 4: c_jail_in is None
data["id"] = str(int(uuid.uuid4().int & (1 << 25)-1))  # can be a new point
data["c_jail_in"] = None  # c_jail_in is None
test_endpoint(data)

# Scenario 5: ID exists in the database
data["id"] = "6108"
test_endpoint(data)

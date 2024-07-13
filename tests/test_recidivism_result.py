import requests
import json

# Define the URL of the endpoint
url = "http://127.0.0.1:5000/recidivism_result/"
# url = 'https://ruipinto-ldsa-production.up.railway.app/recidivism_result/'

# Sample payloads for the your scenarios
payloads = [
    {
        "id": "6108",  # if exists in the database
        "outcome": False  # Different from the predicted outcome
    },
    {
        "id": "999997",  # if id does not exist in the database
        "outcome": False  # Same as the predicted outcome
    },
    {
        "id": None,  # ID that is none
        "outcome": True  # Arbitrary outcome
    }
]

# Set the headers for the request
headers = {
    "Content-Type": "application/json"
}

# Function to send the POST request and print the result


def test_endpoint(payload):
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print(f"Testing with payload: {payload}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)


# Run the tests for the three scenarios
for payload in payloads:
    test_endpoint(payload)

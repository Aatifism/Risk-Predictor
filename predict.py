import requests
import json

FLASK_APP_URL = "http://localhost:5000"  

def make_prediction(user_id):
    url = f"{FLASK_APP_URL}/predict/{user_id}"
    try:
        response = requests.post(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error sending prediction request for user {user_id}: {e}")
        return None

def main():
    user_ids = [1, 6]

    for user_id in user_ids:
        print(f"\nProcessing user {user_id}")
        result = make_prediction(user_id)
        if result:
            print(f"Prediction result: {json.dumps(result, indent=2)}")
        else:
            print(f"Failed to get prediction for user {user_id}")

if __name__ == "__main__":
    main()
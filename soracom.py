import requests

def send_data_to_endpoint(detection_json):
    headers = {'Content-Type': 'application/json'}
    endpoint = 'http://uni.soracom.io'
    try: 
        response = requests.post(endpoint, data=detection_json, headers=headers, timeout=(3.0))
        print(response.json())
    except requests.exceptions.Timeout as e:
        print(e)
        pass
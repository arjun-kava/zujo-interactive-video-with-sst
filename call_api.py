import requests
import json
with open("/media/zujo/Production1/Datasets/Manual-Annotation-Zujo-Fashion-Street2Shop/server/demo3.json", 'r') as f:
    json_data = json.load(f)
print(json_data)
r = requests.post('http://192.168.0.171:8081', json=json_data)
print(r.json())
with open('result.json', 'w') as outfile:
    json.dump(r.json(), outfile)
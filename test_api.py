import json
from fastapi.testclient import TestClient
from pydantic import BaseModel

from .main import app

client = TestClient(app)

class PredData(BaseModel):
    age:int=38
    workclass:str='State-gov'
    fnlgt:int=77516
    education:str='Bachelors'
    education_num:int=13
    marital_status:str='Never-married'
    occupation:str='Adm-clerical'
    relationship:str='Not-in-family'
    race:str='White'
    sex:str='Male'
    capital_gain:int=0
    capital_loss:int=0
    hours_per_week:int=40
    native_country:str='United-States'

post_data = PredData().dict()

def test_get_method():
    r = client.get('/')
    response = json.loads(r.content.decode('utf-8'))
    assert r.status_code == 200
    assert response['message'] == 'Hello Stranger!'

def test_post_method_1():
    r = client.post('/predict', json=post_data)
    response = json.loads(r.content.decode('utf-8'))
    assert r.status_code == 200
    assert response['prediction'][0] == ' <=50K'

def test_post_method_2():
    post_data['capital_gain'] = 20000
    r = client.post('/predict', json=post_data)
    response = json.loads(r.content.decode('utf-8'))
    assert r.status_code == 200
    assert response['prediction'][0] == ' >50K'
    


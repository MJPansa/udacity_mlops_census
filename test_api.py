import json
from pydantic import BaseModel
import pytest
import requests

BASE_URL = 'http://127.0.0.1:8000'

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
    r = requests.get(BASE_URL + '/')
    response = json.loads(r.content.decode('utf-8'))
    assert r.status_code == 200
    assert response['message'] == 'Hello Stranger!'

def test_post_method_1():
    r = requests.post(BASE_URL + '/predict', json=post_data)
    response = json.loads(r.content.decode('utf-8'))
    assert r.status_code == 200
    assert response['prediction'] == ' <=50K'

def test_post_method_2():
    post_data['capital_gain'] = 20000
    r = requests.post(BASE_URL + '/predict', json=post_data)
    response = json.loads(r.content.decode('utf-8'))
    assert r.status_code == 200
    assert response['prediction'] == ' >50K'
    


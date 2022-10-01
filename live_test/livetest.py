import requests
from pydantic import BaseModel

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


if __name__=='__main__':
    r = requests.post('https://census-data-api-udacity.herokuapp.com/predict', json=post_data)
    print(f'status code is: {r.status_code}')
    print(r.content.decode('utf-8'))
# Put the code for your API here.
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


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

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

model = joblib.load('model/random_forest.joblib')



app = FastAPI()



@app.get("/")
async def greeting():
    return {"message": "Hello Stranger!"}


@app.post("/predict")
async def predict(data:PredData) -> dict:
    #transform data into pandas df
    df = pd.DataFrame()
    for key,val in data.dict().items():
        df[key] = [val]
    df.columns = [x.replace('_', '-') for x in df.columns]
    #encode variables
    for cat in cat_features:
        encoder = LabelEncoder()
        encoder.classes_ = np.load(f'encodings/{cat}.npy', allow_pickle=True)
        df[cat] = encoder.transform(df[cat])

    pred = model.predict(df)
    pred_encoder = LabelEncoder()
    pred_encoder.classes_ = np.load(f'encodings/salary.npy', allow_pickle=True)
    pred_encoded = pred_encoder.inverse_transform(pred)

    return {'prediction': list(pred_encoded)}
import pandas as pd
import numpy as np
import os

from ..starter.train_model import fit_encoders, process_data, score_model

df = pd.read_csv('../data/census_clean.csv')

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


def test_fit_encoders():
    label = 'salary'
    encoder_list = fit_encoders(df.copy(), cats=cat_features, label=label, save=False, return_=True)
    for encoder, label_ in zip(encoder_list, cat_features):
        assert len(encoder.classes_) == len(set(df[label_].values))

def test_process_data():
    X_train, y_train, encoder, lb = process_data(
        df, categorical_features=cat_features, label="salary", training=True
    )
    assert isinstance(X_train, pd.DataFrame)
    assert len(X_train) > 0

def test_score_model():
    a = np.array([1,1,1,1,1,1,1,1,0,0])
    b = np.array([1,1,1,1,1,1,1,1,1,1])
    score = score_model(a, b, metric='accuracy')
    assert score==0.8
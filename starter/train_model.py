# Script to train machine learning model.
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if not os.path.exists('../encodings'):
    os.mkdir('../encodings')

# Add the necessary imports for the starter code.

# Add code to load in the data.

# Proces the test data with the process_data function.

def process_data(data, categorical_features, label, training=True):
    encoder_dict = dict()
    if training==True:
        for cat in categorical_features + [label]:
            encoder = LabelEncoder()
            data[cat] = encoder.fit_transform(data[cat])
            np.save(f'../encodings/{cat}.npy', encoder.classes_)
            encoder_dict[cat] = encoder
    elif training==False:
        for cat in categorical_features + [label]:
            encoder = LabelEncoder()
            encoder.classes_ = np.load(f'../encodings/{cat}.npy', allow_pickle=True)
            data[cat] = encoder.transform(data[cat])
            encoder_dict[cat] = encoder

    y_train = data.pop(label)
    X_train = data

    return X_train, y_train, encoder_dict, label
    
    


data = pd.read_csv('../data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

if __name__ == "__main__":

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    joblib.dump(rf, '../model/random_forest.joblib')

    X_test, y_test, encoder, lb = process_data(test, categorical_features=cat_features,
        label='salary', training=False)

    preds = rf.predict(X_test)
    print(f'ACCURACY: {100*accuracy_score(y_test, preds):.2f} %')

    

    



# Train and save a model.

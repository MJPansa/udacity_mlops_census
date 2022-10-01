# Script to train machine learning model.
from typing import List
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

if not os.path.exists('../encodings'):
    os.mkdir('../encodings')

# Add the necessary imports for the starter code.

# Add code to load in the data.

# Proces the test data with the process_data function.

def fit_encoders(data:pd.DataFrame, cats:list, label:str, save=True, return_=False):
    '''
    A utility function to fit the encoders and save 
    encoding params to disk via numpy arrays

    params:
        data: pandas df
        cats: list of categorical cols
        label: label
    '''
    encoder_list = []
    for cat in cats + [label]:
            encoder = LabelEncoder()
            encoder.fit(data[cat])
            if save:
                np.save(f'../encodings/{cat}.npy', encoder.classes_)
            encoder_list.append(encoder)
    if return_:
        return encoder_list


def process_data(data:pd.DataFrame, categorical_features:list, label:str, training:bool=True):
    '''
    A utility function to process the data and return splitted into X and y

    params:
        data: pandas df
        categorical_features: list of features to get encoded
        label: prediction label
        training: indicate whether data is for training 
    '''
    encoder_dict = dict()
    for cat in categorical_features + [label]:
        encoder = LabelEncoder()
        encoder.classes_ = np.load(f'../encodings/{cat}.npy', allow_pickle=True)
        data[cat] = encoder.transform(data[cat])
        encoder_dict[cat] = encoder

    y_train = data.pop(label)
    X_train = data

    return X_train, y_train, encoder_dict, label
    

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

def log_slice_performance(data:pd.DataFrame, preds:np.array, cats:list):
    '''
    utility function that saves slice performance to a local log file.
    we compute the performance for slices on each of the 2 most frequent categorical column values

    params:
        data: pandas df
        preds: numpy array with predictions
        cats: list of categorical columns
    '''
    data['pred'] = preds
    encoder = LabelEncoder()
    encoder.classes_ = np.load(f'../encodings/salary.npy', allow_pickle=True)
    data['salary'] = encoder.transform(data['salary'])

    with open('../performance_logs/slice_acc.txt', 'w') as outfile:
        outfile.writelines('-'*40 + '\n')
        outfile.writelines('-'*40 + '\n')
        outfile.writelines(f'Accuracy scores for slices of the data:\n')
        outfile.writelines('-'*40 + '\n')
        outfile.writelines('-'*40 + '\n\n')
        for cat in cats:
            outfile.writelines(f'feature: {cat}\n')
            class_values = data[cat].value_counts().index[:2]
            for val in class_values:
                slice = data[data[cat]==val]
                acc = accuracy_score(slice.pred.values, slice.salary.values)
                outfile.writelines(f'-> {val}: {100*acc:.2f} %\n')
            outfile.writelines('-'*20 + '\n')

def score_model(y_test, preds, metric:str='accuracy'):
    if metric== 'accuracy':
        score = accuracy_score(y_test, preds)
    elif metric=='precision':
        score = precision_score(y_test, preds)
    return score

if __name__ == "__main__":
    #read data
    print('reading in the data...')
    data = pd.read_csv('../data/census_clean.csv')
    #fit and save encoders
    print('encoding the categorical columns...')
    fit_encoders(data, cat_features, 'salary')
    #split data
    print('splitting the dataset...')
    train, test = train_test_split(data, test_size=0.33)
    #process training data
    print('processing the data...')
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    print('training the model...')
    #create model
    rf = RandomForestClassifier()
    #train and save model
    rf.fit(X_train, y_train)
    print('saving the model...')
    joblib.dump(rf, '../model/random_forest.joblib')
    #process test data
    X_test, y_test, encoder, lb = process_data(test.copy(), categorical_features=cat_features,
        label='salary', training=False)
    #make predictions
    preds = rf.predict(X_test)
    print('model performance metrics:')
    print(f'ACCURACY: {100*score_model(y_test, preds):.2f} %')
    #log slice performance
    log_slice_performance(test, preds, cat_features)
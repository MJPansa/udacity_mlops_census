# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A simple Random Forest Classifier with default parameters trained on census data to predict salary classes

## Intended Use
Use to predict whether someone belongs to class <50k salary or above 50k salary

## Training Data
Official census data

## Evaluation Data
A 33% split of the data 

## Metrics
Eval Metric: Accuracy
Model Performance: 85.84%

## Ethical Considerations
Data slices of the final model need to be further investigated with ethical considerations

## Caveats and Recommendations
RF works solid, try other ML algorithms as well!

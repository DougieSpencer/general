import tensorflow as tf
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import tempfile
import os
import regex
import warnings

warnings.filterwarnings(action='ignore')

dataset = pd.read_csv(r"C:\Users\dougl\Downloads\titanic\train.csv")


def transform_data(dataset):
    transformed_dataset = dataset[["Survived", "Pclass", "Sex", "Age", "Parch", "Fare", "Embarked"]]
    transformed_dataset['Age'].fillna(transformed_dataset['Age'].mean(), inplace=True)
    
    transformed_dataset['Embarked'].fillna(transformed_dataset['Embarked'].value_counts().index[0], inplace=True)
    transformed_dataset['Embarked'] = transformed_dataset['Embarked'].replace("S", 0).replace("C", 0.5).replace("Q", 1)
    transformed_dataset['Sex'] = transformed_dataset['Sex'].replace("male", 0).replace("female", 1)
    
    #sns.swarmplot(transformed_dataset['Fare'])
    
    for column in transformed_dataset.columns:
        transformed_dataset[column] = (transformed_dataset[column] - transformed_dataset[column].min()) / (transformed_dataset[column].max() - transformed_dataset[column].min()) 
    return transformed_dataset

transformed_dataset = transform_data(dataset)

train_dataset = transformed_dataset.sample(frac=0.75, random_state=1)
test_dataset = transformed_dataset.drop(train_dataset.index)

print(train_dataset)



"""
PLAN

Logistic regression variables
normalised pclass
Sex as 0 and 1
normalised age
normalised Fare
normalised cabin
"""
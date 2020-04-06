from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
from azureml.data.datapath import DataPath
import os
import argparse
from sklearn import datasets 
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

#get context when its running in the compute instance
run = Run.get_context()

#read from the input
df = run.input_datasets['train'].to_pandas_dataframe() 

#same code than before
y = df.pop('Class')
X = df

clf = RandomForestClassifier(n_estimators = 100)

cv = cross_validate(clf, X, y, scoring='accuracy', cv=5, n_jobs=-1)
acc = str(round(np.average(cv['test_score']), 3))

#log metrics
run.log(name = 'accuracy', value = acc)

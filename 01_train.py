#%%
from azureml.core import Workspace, Datastore, Dataset, ScriptRunConfig, ComputeTarget, Experiment
from azureml.data.datapath import DataPath
from azureml.train.sklearn import SKLearn
from azureml.train.estimator import Estimator

#%%
#multi tenant with my account 
from azureml.core.authentication import InteractiveLoginAuthentication
int_auth = InteractiveLoginAuthentication(tenant_id='35069d74-1489-4194-80c7-3a81385ead5b')
ws = Workspace.from_config(auth=int_auth)
ws.name

#%%
dataset = Dataset.get_by_name(workspace=ws, name = 'demo_wines_live')

#%%
#point to compute target
comp = ComputeTarget(ws, name = 'compute-instance-demo')
comp 

#%%
#estimator with SKlearn by default + azureml-sdk package
est = SKLearn(
                source_directory='./scripts',
                entry_script='train.py',
                compute_target=comp,
                inputs = [dataset.as_named_input('train')], #readable from the script
                pip_packages=['azureml-sdk', 'pyarrow>=0.12.0']
)

#%%
exp = Experiment(workspace=ws, name = 'submitted_wine')
run = exp.submit(est)
run.wait_for_completion(show_output=True)

#%%
%%writefile ./scripts/train.py
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
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

# %%

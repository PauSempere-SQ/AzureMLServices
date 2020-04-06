from sklearn import datasets 
import pandas as pd
import numpy as np 

X, y = datasets.load_wine(return_X_y=True)
df_to_save = pd.DataFrame(X)

df_to_save['Class'] = y
columns = ['Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 
    'Total Phenols', 'Flavonoids', 'Nonflavonoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280', 'Proline', 'Class'
    ]
df_to_save.columns = columns
df_to_save.to_csv(r'./Data/wine.csv', index=False)

from azureml.core import Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath

#multi tenant with my account 
from azureml.core.authentication import InteractiveLoginAuthentication
int_auth = InteractiveLoginAuthentication(tenant_id='your_tenant_id')
ws = Workspace.from_config(auth=int_auth)
ws.name 

#get the default
ds = ws.get_default_datastore() 
ds 

ds.upload(src_dir = './Data', target_path = 'demo_datasets/tabular/', 
            overwrite=True, show_progress=True)

#get specific datastore 
ds_path = DataPath(datastore=ds, path_on_datastore='demo_datasets/tabular/wine.csv')
ds_path 

#pointing to Azure datapath, not local!
train = Dataset.Tabular.from_delimited_files(path = ds_path)

#pure metadata
print(train)

#register the definition 
train.register(workspace=ws,name='demo_wines_live')


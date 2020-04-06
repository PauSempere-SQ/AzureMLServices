from azureml.core import Workspace, Datastore, Dataset
from azureml.pipeline.core import Pipeline, PublishedPipeline, PipelineEndpoint
from azureml.data.datapath import DataPath
import requests 

#multi tenant with my account 
from azureml.core.authentication import InteractiveLoginAuthentication
int_auth = InteractiveLoginAuthentication(tenant_id='your_tenant_id')
ws = Workspace.from_config(auth=int_auth)
ws.name 

pipe = PipelineEndpoint.get(workspace=ws,name = "Wine_demo_dry_run-batch inference") 

#submit with default parameters and a new experiment name
run = pipe.submit("wine_exp_submitted")
run.wait_for_completion(show_output=True)


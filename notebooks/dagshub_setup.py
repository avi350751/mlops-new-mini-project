import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/avi350751/mlops-new-mini-project.mlflow')

dagshub.init(repo_owner='avi350751', repo_name='mlops-new-mini-project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
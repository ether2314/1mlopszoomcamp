import os
import pickle
import click
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.sklearn.autolog()

mlflow.sklearn.autolog(disable=True)


#MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
#mlflow.set_tracking_uri("file:/home/ether/Desktop/Mini-conda/Project/MLflow/scripts/artifacts")

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Use SQLite as backend for tracking
mlflow.set_experiment("nyc-assignment-experiment")  # Set experiment name


# Function to load pickle file
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)


def run_train(data_path: str):
    
    

    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(rf, "random_forest_model")

        
if __name__ == '__main__':
    run_train()

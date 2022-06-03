import argparse
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('nyc-taxi-experiment')


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):
    
    mlflow.autolog()
    
    with mlflow.start_run():
        mlflow.set_tag('developer', 'poghostick')
        
        mlflow.log_param('train-data-path', './data/green_tripdata_2021-01.parquet')
        mlflow.log_param('val-data-path', './data/green_tripdata_2021-02.parquet')

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mlflow.log_metric('rmse', rmse)
    
    runs = client.search_runs(
        experiment_ids='1',    # Experiment ID we want
        filter_string="metrics.rmse < 7",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse ASC"]
    )
    run_id = runs[0].info.run_id
    print(f"Number of parameters: {len(mlflow.get_run(run_id).to_dictionary()['data']['params'])}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()
    
    run(args.data_path)


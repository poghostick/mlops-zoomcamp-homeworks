import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from pathlib import Path
import pickle
from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from requests import get
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def get_paths(date=None):
    
    logger = get_run_logger()
    
    if not date:
        date = datetime.date.today()
    else:
        date = datetime.datetime.fromisoformat(date)
    train_date = date + relativedelta(months=-2)
    valid_date = date + relativedelta(months=-1)
    train_year, train_month = str(train_date.year), str(train_date.month).zfill(2)
    valid_year, valid_month = str(valid_date.year), str(valid_date.month).zfill(2)

    for year, month in zip([train_year, valid_year], [train_month, valid_month]):
        file_name = f"data/fhv_tripdata_{year}-{month}.parquet"
        if not Path(file_name).is_file():
            logger.info(f"Downloading the file {file_name}...")
            url = ("https://nyc-tlc.s3.amazonaws.com/trip+data/"
                   f"fhv_tripdata_{year}-{month}.parquet")
            with open(file_name, 'wb') as file:
                response = get(url)
                file.write(response.content)
            logger.info("File downloaded")
        else:
            logger.info(f"The file {file_name} already exists. Skipping the download...")
    
    return (f"data/fhv_tripdata_{year}-{month}.parquet" for year, month in zip([train_year, valid_year],
                                                                               [train_month, valid_month]))
    

@task
def prepare_features(df, categorical, train=True):
    
    logger = get_run_logger()
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    
    logger = get_run_logger()
    
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    
    logger = get_run_logger()
    
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow
def main(date=None):
    
    logger = get_run_logger()
    
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    
    if not date:
        date = datetime.date().today().isoformat()
    
    Path('models').mkdir(exist_ok=True)
    dv_loc = f'models/dv-{date}.b'
    with open(dv_loc, 'wb') as dv_out:
        pickle.dump(dv, dv_out)
        logger.info(f"Saving dv into {dv_loc}")
        logger.info(f"File size: {Path(dv_loc).stat().st_size}")


main(date="2021-08-15")
        
        
# DeploymentSpec(
#     name="cron-schedule-deployment",
#     flow=main,
#     schedule=CronSchedule(
#         cron="0 9 15 * *",
#         timezone="Europe/Prague"
#     ),
#     flow_runner=SubprocessFlowRunner(),
# )


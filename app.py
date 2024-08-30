import os
import warnings
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metric(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    csv_url = ("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv")

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Exception occurred:unable to download train and test")
    
    train, test = train_test_split(data)

    X_train = train.drop(['quality'], axis=1)
    X_test = test.drop(['quality'], axis=1)

    y_train = train[['quality']]
    y_test = test[['quality']]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio =  float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

        lr.fit(X_train, y_train)

        predicted_qulities = lr.predict(X_test)

        (rmse, mae, r2) = eval_metric(y_test, predicted_qulities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        remote_server_uri = ""
        mlflow.set_tracking_uri(remote_server_url)

        tracking_url_type_store = urlparse( mlflow.get_tracing_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")



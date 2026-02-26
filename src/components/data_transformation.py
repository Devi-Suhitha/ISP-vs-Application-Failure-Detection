import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):

        try:
            numeric_features = [
                "ping_latency_ms",
                "packet_loss_percent",
                "jitter_ms",
                "download_speed_mbps",
                "dns_time_ms",
                "initial_buffer_time_sec",
                "buffering_events",
                "avg_bitrate_kbps",
                "resolution",
                "segment_download_time_ms",
                "server_cpu_percent",
                "server_memory_percent",
                "error_rate_5xx_percent",
                "db_response_time_ms",
                "peak_hour_flag"
            ]

            categorical_features = [
                "activity_type",
                "isp_name",
                "region",
                "device_type"
            ]

            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_features),
                ("cat", cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "root_cause"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessor = self.get_preprocessor()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            return X_train_transformed, X_test_transformed, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
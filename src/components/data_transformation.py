import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException

class DataTransformation:

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "root_cause"

            numerical_columns = [
                "ping_latency_ms",
                "packet_loss_percent",
                "download_speed_mbps",
                "server_cpu_percent",
                "error_rate_5xx_percent",
                "db_response_time_ms",
                "initial_buffer_time_sec",
            ]

            scaler = StandardScaler()

            X_train = scaler.fit_transform(train_df[numerical_columns])
            X_test = scaler.transform(test_df[numerical_columns])

            y_train = train_df[target_column]
            y_test = test_df[target_column]

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    try:
        logging.info("Training pipeline started")

        # 🔹 Step 1: Data Ingestion
        ingestion_obj = DataIngestion()
        train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()

        logging.info("Data ingestion completed")

        # 🔹 Step 2: Data Transformation
        data_transformation_obj = DataTransformation()

        train_array, test_array, y_train, y_test = (
            data_transformation_obj.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
        )

        logging.info("Data transformation completed")

        # 🔹 Step 3: Model Training
        model_trainer_obj = ModelTrainer()

        accuracy = model_trainer_obj.initiate_model_trainer(
            train_array,
            test_array,
            y_train,
            y_test
        )

        logging.info("Model training completed")
        print(f"Model Accuracy: {accuracy}")

    except Exception as e:
        raise CustomException(e, sys)
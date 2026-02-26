import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.exception import CustomException
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):

        try:
            # Convert labels to numeric for XGBoost
            y_train = y_train.map({"ISP_SIDE": 0, "APP_SIDE": 1})
            y_test = y_test.map({"ISP_SIDE": 0, "APP_SIDE": 1})

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1]
            }

            xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                
                random_state=42
            )

            grid_search = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            save_object(self.config.trained_model_file_path, best_model)

            print("Best Parameters:", grid_search.best_params_)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
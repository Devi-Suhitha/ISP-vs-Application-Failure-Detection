import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.utils import save_object
from src.exception import CustomException

class ModelTrainer:

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            save_object("artifacts/model.pkl", model)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
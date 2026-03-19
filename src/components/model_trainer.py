import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            best_model_score = max(model_report.values())
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("No best model found with R2 score greater than 0.6")
                raise CustomException("No best model found with R2 score greater than 0.6", sys)  # type: ignore
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            # model_list = []
            # r2_list = []

            # for model_name, model in models.items():
            #     model.fit(X_train, y_train)
            #     y_pred = model.predict(X_test)
            #     r2_score_value = r2_score(y_test, y_pred)
            #     model_list.append(model_name)
            #     r2_list.append(r2_score_value)

            # best_model_score = max(r2_list)
            # best_model_index = r2_list.index(best_model_score)
            # best_model_name = model_list[best_model_index]
            # best_model = models[best_model_name]

            logging.info(f"Best Model Found: {best_model_name} with R2 Score: {best_model_score}")

            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            logging.info(f"R2 Score for the best model: {r2_score_value}")

            return r2_score_value

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj=best_model
            # )

        except Exception as e:
            logging.info("Exception occurred during model training")
            raise CustomException(e, sys)  # type: ignore

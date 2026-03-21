import os
import pickle
import sys

from sklearn.model_selection import GridSearchCV

from src.logger import logging
from sklearn.metrics import r2_score
from src.exception import CustomException
import pandas as pd
import dill

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        models_list=list(models.values())
        model_keys=list(models.keys())
        for i in range(len(models)):
            model = models_list[i]
           
            para=param[model_keys[i]]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
           
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_keys[i]] = test_model_score

        return report

    except Exception as e:
        logging.info("Exception occurred during model evaluation")
        raise CustomException(e, sys)  # type: ignore

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) # type: ignore
    

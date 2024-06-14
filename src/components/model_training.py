import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_models
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'gausianNB':GaussianNB(var_smoothing=0.5),
            'random_forest':RandomForestClassifier(n_estimators=100,criterion='gini',random_state=0),
            'gradiant_boost':GradientBoostingClassifier(),
            'k_nearest_naibour':KNeighborsClassifier(),
            'xgboost':XGBClassifier(),
                    }
            
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , acuracy score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , acuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

            return best_model , best_model_name , best_model_score
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
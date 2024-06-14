import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import download_model, load_object
import pandas as pd

from flask import Request

from dataclasses import dataclass


@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str="predictions"
    prediction_file_name: str ="predicted_file.csv"
    model_file_path: str = os.path.join(artifact_folder, "model.pkl")
    prepocessor_path: str = os.path.join(artifact_folder, "preprocessor.pkl")
    prediction_file_path: str =os.path.join(prediction_output_dirname, prediction_file_name)


class PredicttionPipeline:
    def __init__(self, request:Request):

        self.request = request
        self.prediction_file_detail =PredictionFileDetail()


    def save_input_files(self) -> str:
        try:
            pred_file_input_dir="prediction_artifacts"
            os.makedirs(pred_file_input_dir,exist_ok=True)

            input_csv_file =self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir,input_csv_file)

            input_csv_file.save(pred_file_path)


            return pred_file_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def predict(self, features):
        try:
            model = self.utils.load_object(self.prediction_file_detail.model_file_path)
            preprocessor = self.utils.load_object(file_path=self.prediction_file_detail.prepocessor_path)

            transformed_x = preprocessor.transform(features)
            preds = model.predict(transformed_x)

            logging.info("uploaded file  prediction setup complited")

            return preds
        
        
        except Exception as e:
            raise CustomException(e,sys)
        


    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):
        try:
            prediction_column_name : str = TARGET_COLUMN
            input_dataframe = pd.read_csv(input_dataframe_path)

            input_dataframe = input_dataframe.drop(columns="default payment next month") if "default payment next month" in input_dataframe.columns else input_dataframe
            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'default next m', 1:'will pay next m'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)

            os.makedirs(self.prediction_file_detail.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.prediction_file_detail.prediction_file_path, index=False)

            logging.info("prediction completed")
        
        except Exception as e:
            raise CustomException(e,sys)

        

    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_file_detail
        
        except Exception as e:
            raise CustomException(e,sys)

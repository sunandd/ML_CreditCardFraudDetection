import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import download_model, load_object

from flask import request

from dataclasses import dataclass


@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str="predictions"
    prediction_file_name: str ="predicted_file.csv"
    prediction_file_path: str =os.path.join(prediction_output_dirname, prediction_file_name)


class PredicttionPipeline:
    def __init__(self, request:request):

        self.request = request
        self.prediction_file_detail =PredictionFileDetail()


    def save_input_files(self) -> str:
        try:
            pred_file_input_dir="prediction_artifacts"
            os.makedirs(pred_file_input_dir,exist_ok=True)

            input_csv_file =self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir,input_csv_file)

            input_csv_file_save(pred_file_path)


            return pred_file_path
        except Exception as e:
            raise CustomException(e,sys)
        
        

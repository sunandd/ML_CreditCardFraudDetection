import os
import sys
from flask import Flask , render_template, request,jsonify, send_file
from src.exception import CustomException
from src.logger import logging

from src.pipelines.training_pipeline import TrainPipeline
from src.pipelines.prediction_pipeline import PredicttionPipeline


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return "Training Completed"
    
    except Exception as e:
        raise CustomException(e,sys)
    
@app.route('/predict', methods=['POST' , 'GET'])
def predict_route():

    try:
        if request.method == 'POST':
            #its  object of prediction pipeline
            prediction_pipeline=PredicttionPipeline(request)

            prediction_file_summery= prediction_pipeline.run_pipeline()
            logging.info("prediction completed")

            return  send_file(prediction_file_summery.prediction_file_path,
                              download_name= prediction_file_summery.prediction_file_name,
                               as_attachment=True )
        
        else:
            return render_template("upload.html")
        
    except Exception as e:
        raise CustomException(e, sys)    

    





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug = True)
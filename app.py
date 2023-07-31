import os
import sys
from flask import Flask , render_template, request,jsonify, send_file
from src.exception import CustomException

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
    





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug = True)
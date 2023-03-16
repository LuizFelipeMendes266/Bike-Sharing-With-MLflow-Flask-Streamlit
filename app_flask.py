from flask import Flask, request
import mlflow
import pandas as pd


app=Flask(__name__)

@app.route("/bike_sharing", methods=['POST'])
def index():
    data = request.get_json()
    data = data['data']
    df = pd.DataFrame.from_dict(data)

    mlflow.set_experiment('Prediction Bike')

    mlflow.search_runs()
    
    last_run = dict(mlflow.search_runs().sort_values(by='start_time',ascending=False).iloc[0])
    artifact = last_run['artifact_uri']


    model = mlflow.sklearn.load_model(artifact+"/model_pipeline")
    
    predictions = model.predict(df)

    return " Bike Share Demand prediction {}".format(predictions[0])





if __name__=='__main__':
    app.run()
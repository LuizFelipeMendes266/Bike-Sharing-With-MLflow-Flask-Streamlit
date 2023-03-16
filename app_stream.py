import streamlit as st
import mlflow
import pandas as pd


# Define a function to load the MLflow model
def load_mlflow_model():

    mlflow.set_experiment('Prediction Bike')
    mlflow.search_runs()
    last_run = dict(mlflow.search_runs().sort_values(by='start_time', ascending=False).iloc[0])
    artifact = last_run['artifact_uri']
    model = mlflow.sklearn.load_model(artifact + "/model_pipeline")
    return model


# Define the Streamlit app
def app():

    # Load the MLflow model
    model = load_mlflow_model()
    st.set_page_config(page_title="Bike Sharing Demand Predictions", page_icon="🚲",layout='wide')
    st.title("Bike Sharing Demand Predictions")
    st.write("This app predicts the number of bikes rented per hour based on various features such as season, weather, temperature, and more. Simply fill out the form on the left and hit the 'Predict' button to see the predicted number of bike rentals for a given hour.")

    # Define the feature inputs
    season = st.sidebar.selectbox('Season', ['Spring', 'Summer', 'Fall', 'Winter'])
    holiday = st.sidebar.selectbox('Holiday', ['No', 'Yes'])
    workingday = st.sidebar.selectbox('Working Day', ['No', 'Yes'])
    weather = st.sidebar.selectbox('Weather', ['Clear or Few clouds', 'Mist or Cloudy', 'Light Rain', 'Heavy Rain or Snow'])

    temp = st.sidebar.number_input('Temperature', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    atemp = st.sidebar.number_input('Apparent Temperature', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.sidebar.number_input('Humidity', min_value=0, max_value=100, value=50, step=1)
    windspeed = st.sidebar.number_input('Wind Speed', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    hour = st.sidebar.number_input('Hour', min_value=0, max_value=23, value=12, step=1)
    month = st.sidebar.number_input('Month', min_value=1, max_value=12, value=6, step=1)
    weekday = st.sidebar.number_input('Weekday', min_value=0, max_value=6, value=3, step=1)

    map = {
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Winter": 4,
    "No": 0,
    "Yes": 1,
    "Clear or Few clouds": 1,
    "Mist or Cloudy": 2,
    "Light Rain": 3,
    "Heavy Rain or Snow": 4
    }

    # Transform input values to output values using the dictionary
    
    season = map[season]
    holiday = map[holiday]
    workingday = map[workingday]
    weather = map[weather]

    # Define the prediction button
    if st.button('Predict'):
        # Convert the feature inputs to a dataframe
        features_df = pd.DataFrame({
            'season': [season],
            'holiday': [holiday],
            'workingday': [workingday],
            'weather': [weather],
            'temp': [temp],
            'atemp': [atemp],
            'humidity': [humidity],
            'windspeed': [windspeed],
            'hour': [hour],
            'month': [month],
            'weekday': [weekday]
        })

        st.write(features_df)
        # Perform the prediction using the loaded MLflow model
        prediction = model.predict(features_df)

        # Display the prediction
        prediction_formatted = "{:,.2f}".format(float(prediction[0])).replace(",", ".")

        st.success(f'The predicted demand is {prediction_formatted} bikes.')

# Run the Streamlit app
if __name__ == "__main__":
    app()

from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from datetime import datetime

# Load the trained pipeline
model_pipeline = joblib.load('sales_model_pipeline.pkl')

# Initialize Flask application
app = Flask(__name__)

# Function to preprocess input data
def preprocess_input(data):
    # Convert Date to datetime and extract date features
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday
    # Drop original Date column
    data = data.drop(columns=['Date'])
    return data

# Route for home page to render input form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        input_data = {
            "QuantitySold": float(request.form['QuantitySold']),
            "UnitPrice": float(request.form['UnitPrice']),
            "Date": request.form['Date'],
            "ProductCategory": request.form['ProductCategory'],
            "PaymentMethod": request.form['PaymentMethod'],
            "StoreID": int(request.form['StoreID'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        input_df = preprocess_input(input_df)

        # Predict the TotalAmount using the model
        prediction = model_pipeline.predict(input_df)
        
        # Render the result in the result.html template
        return render_template('result.html', prediction=round(prediction[0], 2))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    
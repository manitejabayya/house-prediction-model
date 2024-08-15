from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv("final_dataset.csv")
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                              columns=['beds', 'baths', 'size', 'zip_code'])
    print("Input Data:")
    print(input_data)

    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])
    print("Processed Input Data:")
    print(input_data)

    prediction = pipe.predict(input_data)[0]

    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)


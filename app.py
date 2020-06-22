import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import pickle

# Load Model
model = pickle.load(open('model.pkl','rb'))
model_columns = pickle.load(open('model_columns.pkl','rb'))
# App
app = Flask(__name__)

# Define routes
@app.route('/', methods=['POST'])
def predict():

	data = request.get_json(force=True)
	# convert input data into dataframe
	data.update((x, [y]) for x,y in data.items())
	data_df = pd.DataFrame.from_dict(data)
	data_final = pd.get_dummies(data_df, drop_first=True)

	for col in model_columns:
		if col not in data_final.columns:
			data_final[col] = 0
    
    
	# predictions
	result = model.predict(data_final) # age,JobType,EdType,maritalstatus,occupation,relationship,race,gender,capitalgain,capitalloss,hoursperweek,nativecountry
    
	# send result back to browser
	output = {'results': int(result[0])}

	# return data
	return jsonify(results=output)

if __name__ == '__main__':
    app.run(debug=True)	



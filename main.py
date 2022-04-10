
from flask import Flask
from flask import request
app = Flask(__name__)

import pandas as pd
from pycaret.regression import load_model, predict_model

#. Load trained Pipeline
model = load_model('catboostV1pm25')



from image_features_extractor import extract_features_from_fullimage
import numpy as np
import cv2

@app.route('/api/aqi', methods=['POST'])
def post__api_aqi():
    #cuttle-environment-set-config aqi-pred method=POST route=/api/aqi response=output
    
    file = request.files['file']
    file_string = file.read()
    npimg = np.fromstring(file_string, np.uint8)    
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    train_features=extract_features_from_fullimage(img)
    datatest=pd.DataFrame(train_features)
    unseen_predictions = predict_model(model, data=datatest)
    output=str(unseen_predictions.Label[0])
    
    return output


if __name__ == '__main__':
    app.run()
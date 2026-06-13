from fastapi import FastAPI,Security,HTTPException,Depends
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
from fastapi.security import APIKeyHeader
import os
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title = 'Churn Prediction')

load_dotenv()

api_key = 'X-API-Key'
api_key_header = APIKeyHeader(name=api_key,auto_error=False)
password = os.getenv('apiKey')

def get_api_key(api_key:str = Security(api_key_header)):
    if api_key == password:
        return api_key
    raise HTTPException(status_code=401,detail='Auth Failed')


model = joblib.load('models/churn_model.pkl')
onehot_encoder = joblib.load('models/onehot_encoder.pkl')
scaler = joblib.load('models/scaler.pkl')


class Customer(BaseModel):
     contract : Literal ['Month-to-month', 'Two year', 'One year']
     payment_method : Literal ['Mailed check','Electronic check','Bank transfer (automatic)','Credit card (automatic)']
     internet_service : Literal ['DSL', 'Fiber optic', 'No']
     online_security : Literal ['Yes', 'No', 'No internet service']
     online_backup : Literal ['Yes', 'No', 'No internet service']
     tech_support : Literal ['No', 'Yes', 'No internet service']
     monthly_charges : float = Field(...,gt=0)
     tenure_months : int = Field(...,gt=0)


@app.post('/predict', dependencies=[Depends(get_api_key)])
def predict_churn(customer : Customer):

    input = pd.DataFrame(
        {
            'Contract' : [customer.contract],
            'Payment Method' : [customer.payment_method],
            'Internet Service' : [customer.internet_service],
            'Online Security' : [customer.online_security],
            'Online Backup' : [customer.online_backup],
            'Tech Support' : [customer.tech_support],
            'Monthly Charges' : [customer.monthly_charges],
            'Tenure Months' : [customer.tenure_months]
        }
    )

    categorical_features = ['Contract','Payment Method','Internet Service',
                         'Online Security','Online Backup','Tech Support']
    
    numerical_features = ['Monthly Charges','Tenure Months']

    encoded = onehot_encoder.transform(input[categorical_features])
    scaled = scaler.transform(input[numerical_features])

    final_input = np.concatenate([encoded,scaled] , axis=1)

    prediction = model.predict(final_input)

    return {'Churn' : 'Yes' if prediction[0] == 1 else 'No'}
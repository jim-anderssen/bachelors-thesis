from fastapi import FastAPI
from joblib import load
import numpy as np
from pydantic import BaseModel



class SampleInput(BaseModel):
    sample: list[float]

app = FastAPI()

model_dict = {'CaO':load('../models_scalers/ca_model.joblib'),
              'SiO2':load('../models_scalers/si_model.joblib'),
              'Al2O3':load('../models_scalers/al_model.joblib'),
              'Fe2O3':load('../models_scalers/fe_model.joblib')
             }

scaler_dict = {'CaO':load('../models_scalers/ca_scaler.joblib'),
               'SiO2':load('../models_scalers/si_scaler.joblib'),
               'Al2O3':load('../models_scalers/al_scaler.joblib'),
               'Fe2O3':load('../models_scalers/fe_scaler.joblib')
              }

@app.post('/predict')
async def predict(input_data: SampleInput):
    try:
        result = {}
        sample = np.array(input_data.sample, dtype=np.float64)
        
        for key, model in model_dict.items():
            result[key] = predict_with_each_model(model, scaler_dict[key], sample)
            
        return {'predictions': result}
    except Exception as e:
        return {'error': str(e)}

def predict_with_each_model(model, scaler, sample):
    try:
        scaled_sample = scaler.transform(sample.reshape(1, -1))
        y_pred = model.predict(scaled_sample)
        return round(float(y_pred[0]),2)  # Convert numpy float to Python float
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    
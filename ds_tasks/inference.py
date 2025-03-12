import pandas as pd

def request_model_batch(model, input_data):    
    # let us get the predictions using the classifier we had fit above
    predictions = model.predict(input_data)
    print(f"{sum(predictions)} / {len(input_data)} people with diabetes predicted")
    return pd.DataFrame(predictions)


def request_model_single(model, input_data):
    print("Prediction being made")
    df_predict = pd.DataFrame(input_data)
    prediction = model.predict(df_predict)
    
    diabetes = False
    if sum(prediction) >= 1.0:
        diabetes = True
        
    print(f"Predicted diabetes: {diabetes}")

    return diabetes


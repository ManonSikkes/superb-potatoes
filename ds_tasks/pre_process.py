import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(diabetes_data):
    # The data contains some zero values which make no sense (like 0 skin thickness or 0 BMI). 
    # The following columns/variables have invalid zero values:
    # glucosem bloodPressure, SkinThicknes, Insulin and BMI
    # We will replace these zeros with NaN and after that we will replace them with a suitable value.
    
    print("Imputing missing values")
    # Replacing with NaN
    diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
    # Imputing NaN
    diabetes_data['Glucose'].fillna(diabetes_data['Glucose'].mean(), inplace = True)
    diabetes_data['BloodPressure'].fillna(diabetes_data['BloodPressure'].mean(), inplace = True)
    diabetes_data['SkinThickness'].fillna(diabetes_data['SkinThickness'].median(), inplace = True)
    diabetes_data['Insulin'].fillna(diabetes_data['Insulin'].median(), inplace = True)
    diabetes_data['BMI'].fillna(diabetes_data['BMI'].median(), inplace = True)
    return diabetes_data

def prepare_for_training(diabetes_data):
    X = diabetes_data.drop(["Outcome"], axis = 1) 
    y = diabetes_data.Outcome
    return X, y

def scale_data(X):
    print("Scaling data")
    # Since we are using a distance metric based algorithm we will use scikits standard scaler to scale all the features to [-1,1]
    sc_X = StandardScaler()
    X = pd.DataFrame(sc_X.fit_transform(X,),
            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age'])
    return X

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

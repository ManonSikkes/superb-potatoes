from pre_process import *
from train import *
from load import *
from inference import *

data_file_name = "diabetes.csv"
model_file_name = "knn.joblib"
model_name = f"knn_diabetes_classifier"

X_train_fn = f'X_train_{model_name}.csv'
X_test_fn = f'X_test_{model_name}.csv'
y_train_fn = f'y_train_{model_name}.csv'
y_test_fn = f'y_test_{model_name}.csv'
y_pred_fn = f'y_pred_{model_name}.csv'

# --------------------------------------------------------------------------------- #
# Steps for data pre-processing
# --------------------------------------------------------------------------------- #

# ingest
content = ingest_data()

# load
write_to_csv(data_file_name, content)
df = load_dataframe(data_file_name)

# pre_process
df = clean_data(df)
X, y = prepare_for_training(df)
X = scale_data(X)

# split
X_train, X_test, y_train, y_test = split_data(X, y)

# load
write_df_to_file(X_train, X_train_fn, header = True)
write_df_to_file(X_test, X_test_fn, header = True)
write_df_to_file(y_train, y_train_fn, header = False)
write_df_to_file(y_test, y_test_fn, header = False)

# --------------------------------------------------------------------------------- #
# Steps for training
# --------------------------------------------------------------------------------- #

# ingest
X_train = load_dataframe(X_train_fn)
y_train = load_dataframe(y_train_fn, header = None)

# train
knn = train_model(X_train, y_train)

# load
write_model_to_file(model_file_name, knn)

# --------------------------------------------------------------------------------- #
# Steps for batch inference
# --------------------------------------------------------------------------------- #

# ingest
model = load_model_from_file(model_file_name)
X_test = load_dataframe(X_test_fn)

# predict
y_pred = request_model_batch(model, X_test)

# load
write_df_to_file(y_pred, y_pred_fn, header=['diabetes_prediction'], index_label='index')

# --------------------------------------------------------------------------------- #
# Steps for evaluation
# --------------------------------------------------------------------------------- #

# ingest
model = load_model_from_file(model_file_name)
X_test = load_dataframe(X_test_fn)
y_test = load_dataframe(y_test_fn, header = None)
y_pred = load_dataframe(y_pred_fn)

# evaluate
evaluate_model(model, X_test, y_test, y_pred)

# --------------------------------------------------------------------------------- #
# Steps for single inference
# --------------------------------------------------------------------------------- #

# ingest
model = load_model_from_file(model_file_name)

input_data = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}

diabetes = request_model_single(model, input_data)

result = {
    "diabetes?": diabetes, "input_data": input_data
}

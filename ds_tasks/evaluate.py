from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, y_pred):
    # Get accuracy on test set. Note: In case of classification algorithms score method represents accuracy.
    score = model.score(X_test, y_test)
    print('KNN accuracy: ' + str(score))
            
    # Output classification report
    print('Classification report:')
    print(classification_report(y_test, y_pred))

# Evaluating is inferring predictions with the known Y
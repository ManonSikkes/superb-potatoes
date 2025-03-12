from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def train_model(X_train, y_train):
    
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=7) 
    
    # Fit the model on training data
    knn.fit(X_train, y_train)

    return knn

def evaluate_model(knn, X_test, y_test):
    
    # Get accuracy on test set. Note: In case of classification algorithms score method represents accuracy.
    score = knn.score(X_test, y_test)
    print('KNN accuracy: ' + str(score))
    
    # let us get the predictions using the classifier we had fit above
    y_pred = knn.predict(X_test)
            
    # Output classification report
    print('Classification report:')
    print(classification_report(y_test, y_pred))

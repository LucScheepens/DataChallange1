from sklearn.metrics import confusion_matrix
import numpy as np

def create_confusion_matrix(retinopathy_model, Y_test, X_test):
    """"
    Function to create a confusion matrix
    """

    y_pred = retinopathy_model.predict_classes(X_test, verbose = 1)

    #convert to labels
    y_test = np.where(Y_test==1)[1]

    confusion_numbers = confusion_matrix(y_test, y_pred)
    print(confusion_numbers)
    return confusion_numbers
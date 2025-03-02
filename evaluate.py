from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).argmax(axis=1)
    report = classification_report(y_test, y_pred)
    print(report)

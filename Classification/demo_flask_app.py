from flask import Flask, jsonify
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

@app.route("/api/fraud-detection/health.json")
def health():
    return jsonify({"status": "UP"}), 200


@app.route("/api/fraud-detection/model")
def get_model():
    """dataset = pd.read_csv('PS_20174392719_1491204439457_log.csv')"""
    dataset = pd.read_csv('PS_Sample_log.csv')
    
    X = dataset.iloc[:, [2, 4, 5, 7, 8]].values
    y = dataset.iloc[:, 9].values
      
    y_target = dataset['isFraud']
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify=y_target)
    
    # Training the Logistic Regression model on the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    """score = classifier.score(X_test, y_test)
    print(score)"""
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    """print(cm)"""
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    """print("True Negatives: ",tn)
    print("False Positives: ",fp)
    print("False Negatives: ",fn)
    print("True Positives: ",tp)"""
    
    from sklearn import metrics
    print("Accuracy:     ", round(metrics.accuracy_score(y_test, y_pred),4)*100)
    print("Precision:    ", round(metrics.precision_score(y_test, y_pred),4)*100)
    print("Recall:       ", round(metrics.recall_score(y_test, y_pred),4)*100)
    
    accuracy = round(metrics.accuracy_score(y_test, y_pred),4)*100
    precision = round(metrics.precision_score(y_test, y_pred),4)*100
    recall = round(metrics.recall_score(y_test, y_pred),4)*100
    
    result = [
        {
            'Test data Size': y_test.size,
            'Non-Fradulent predicted True': tn.item(),
            'Non-Fradulent predicted false': fn.item(),
            'Fradulent predicted True': tp.item(),
            'Fradulent predicted false': fp.item()
        },
        
        {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        }
    ]
    return jsonify(result), 200


@app.route("/api/fraud-detection")
def index():
    return "Welcome to Python Server"

if __name__ == "__main__":
    # app.config['DEBUG'] = True
    app.run(host='localhost', port='8090')
    # app.run()
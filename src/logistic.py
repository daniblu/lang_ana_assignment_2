import os
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from joblib import dump

def input_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="data pkl file name")
    args = parser.parse_args()

    return(args)

def main(data):
    
    # load data
    X_train_feats, X_test_feats, y_train, y_test = pd.read_pickle(os.path.join("..", "models", data))

    # initialise classifier and fit model
    classifier = LogisticRegression().fit(X_train_feats, y_train)

    # make predictions
    y_pred = classifier.predict(X_test_feats)

    # make classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)

    # strip file extension from data filename
    data = data.replace(".pkl", "")
    
    # save model
    dump(classifier, os.path.join("..", "models", f"log_{data}.joblib"))
    
    # save report to txt file
    txtpath = os.path.join("..", "out", f"log_report_{data}.txt")
    txtfile = open(txtpath, "w")
    txtfile.write(classifier_metrics)
    txtfile.close()

if __name__ == "__main__":
    args = input_parse()
    main(args.data)
import os
import pandas as pd
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from joblib import dump

def input_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="data pkl file name")
    parser.add_argument("--hidden_layers", nargs='+', type=int, required=True, help="hidden layers design, e.g., 20 10 for two hidden layers with respective number of units")
    parser.add_argument("--max_iter", type=int, default=1000, help="max number of iterations, default=1000")
    parser.add_argument("--random_state", type=int, default=1, help="random state for model training, default=1")
    args = parser.parse_args()

    return(args)

def main(data, hidden_layers, max_iter, random_state):
    
    # load data
    X_train_feats, X_test_feats, y_train, y_test = pd.read_pickle(os.path.join("..", "models", data))

     # create tuple from hidden_layers argument
    hidden_layers_tuple = tuple(hidden_layers)
    
    # initialise classifier and fit model
    classifier = MLPClassifier(activation = "logistic",
                            hidden_layer_sizes = hidden_layers_tuple,
                            max_iter=max_iter,
                            random_state = random_state).fit(X_train_feats, y_train)

    # make predictions
    y_pred = classifier.predict(X_test_feats)

    # make classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)

    # strip file extension from data filename
    data = data.replace(".pkl", "")
    
    # save model
    modelpath = os.path.join("..", "models", f"nn_{data}")
    dump(classifier, f"{modelpath}.joblib")

    # save txt file with model parameter values
    txtpath = os.path.join(f"{modelpath}.txt")
    txtfile = open(txtpath, "w")
    L = [f"Hidden layers: {hidden_layers_tuple} \n", 
         f"Max iterations: {max_iter} \n", 
         f"Random state: {random_state}"]
    txtfile.writelines(L)
    txtfile.close()
    
    # save report to txt file
    txtpath = os.path.join("..", "out", f"nn_report_{data}.txt")
    txtfile = open(txtpath, "w")
    txtfile.write(classifier_metrics)
    txtfile.close()

if __name__ == "__main__":
    args = input_parse()
    main(data=args.data, hidden_layers=args.hidden_layers, max_iter=args.max_iter, random_state=args.random_state)
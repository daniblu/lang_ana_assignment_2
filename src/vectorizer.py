# libraries
import os
import argparse
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser(description="Vectorize data with user-determined parameters. Saves vectorizer and transformed data.")
    # add arguments
    parser.add_argument("--vecid", type=int, required=True, help="unique int to identify vectorizer")
    parser.add_argument("--split", type=float, required=True, help="proportion of test data")
    parser.add_argument("--random_state", type=int, default=1, help="random state for data splitting, default=1")
    parser.add_argument("--tfidf", action="store_true", help="apply tfidf")
    parser.add_argument("--ngrams", nargs='+', type=int, required=True, help="features of interest, e.g., 1 2 for unigrams and bigrams")
    parser.add_argument("--max", type=float, help="upper limit to document frequency, e.g., 0.95")
    parser.add_argument("--min", type=float, help="lower limit to document frequency, e.g., 0.05")
    parser.add_argument("--n", type=int, required=True, help="number of features")
    #parse the arguments from the command line
    args = parser.parse_args()
        
    return(args)

def main(vecid, split, random_state, tfidf, ngrams, max, min, n):
    # load data
    filepath = os.path.join("..", "in", "fake_or_real_news.csv")
    data = pd.read_csv(filepath, index_col=0)
    
    # isolate texts and labels
    X = data["text"]
    y = data["label"]
    
    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X,                   # texts for the model
                                                        y,                   # classification labels
                                                        test_size=split,     # split proportion
                                                        random_state=random_state) # random state for reproducibility

    # create tuple from ngrams argument
    ngrams_tuple = tuple(ngrams)

    # initialise vectorizer
    if tfidf:
        vec_method = "tfidf"
        vectorizer = TfidfVectorizer(ngram_range = ngrams_tuple,
                                    lowercase =  True,
                                    max_df = max,
                                    min_df = min,
                                    max_features = n)
    
    else:
        vec_method = "count"
        vectorizer = CountVectorizer(ngram_range = ngrams_tuple,
                                    lowercase =  True,
                                    max_df = max,
                                    min_df = min,
                                    max_features = n)
    
    # fit vectorizer and transform/vectorize data
    X_train_feats = vectorizer.fit_transform(X_train)
    
    # transform test data
    X_test_feats = vectorizer.transform(X_test)
    
    # save vectorizer with unique filename
    vecpath = os.path.join("..", "models", f"vect{vecid}")
    dump(vectorizer, f"{vecpath}.joblib")

    # save txt file with vectorizer parameter values
    txtpath = os.path.join(f"{vecpath}.txt")
    txtfile = open(txtpath, "w")
    L = [f"Vectorizer method: {vec_method} \n", 
         f"Number of features: {n} \n", 
         f"N-grams: {ngrams} \n",
         f"Max document frequency: {max} \n",
         f"Min document frequency: {min} \n",
         f"Random state: {random_state}"]
    txtfile.writelines(L)
    txtfile.close()

    # save transformed data
    data = [X_train_feats, X_test_feats, y_train, y_test]    
    
    datafilname = f"split{split}_vect{vecid}.pkl"
    datapath = os.path.join("..", "models", datafilname)
    
    with open(datapath, 'wb') as file:
        pd.to_pickle(data, datapath)

if __name__ == "__main__":
    args = input_parse()
    main(vecid=args.vecid, split=args.split, random_state=args.random_state, tfidf=args.tfidf, ngrams=args.ngrams, max=args.max, min=args.min, n=args.n)
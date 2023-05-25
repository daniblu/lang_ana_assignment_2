# language analytics assignment 2
This repository is assignment 2 out of 5, to be sumbitted for the exam of the university course [Language Analytics](https://kursuskatalog.au.dk/en/course/115693/Language-Analytics) at Aarhus Univeristy.

The first section describes the assignment task as defined by the course instructor. The section __Student edit__ is the student's description of how the repository solves the task and how to use it.

## Text classification benchmarks

This assignment is about using ```scikit-learn``` to train simple (binary) classification models on text data. For this assignment, we'll continue to use the Fake News Dataset that we've been working on in class.

For this exercise, you should write *two different scripts*. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:

- Be executed from the command line
- Save the classification report to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

## Objective

This assignment is designed to test that you can:

1. Train simple benchmark machine learning classifiers on structured text data;
2. Produce understandable outputs and trained models which can be reused;
3. Save those results in a clear way which can be shared or used for future analysis

## Student edit
### Solution
The code written for this assignment can be found within the ```src``` directory. The directory contains three scripts, all with arguments that can be set from the terminal. Here follows a description of the funcionality of each script:

- __vectorizer.py__: Takes the ```fake_or_real_news.csv``` from ```in```, vectorizes the text and splits the data into train and test sets according to parameter values set by the user. See ```python3.9 vectorizer.py -h``` for an overview of manipulatable parameters and user instructions. The script outputs a vectorizer model, a text file indicating the parameter values of the vectorizer, and a ``pkl``-file with the vectorized data, all saved in ```models```. All filenames incorporate a user-defined ID integer to distinguish among vectorizers produced from different runs of the script.

- __logistic.py__: Takes a data ``pkl``-file of choice and performs logistic regression on the data. The script offers the user no influence on the parameters of the logistic regression. All parameter values are set to their default for handling binary data (see documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)). The script outputs a fitted logistic model in the ```models``` directory and a classification report text file in ```out```.

- __neuralnet.py__: Takes a data ``pkl``-file of choice and builds a neural network classification model based on the data. See ```python3.9 neuralnet.py -h``` for an overview of manipulatable parameters. The script outputs a fitted model and a text file indicating the parameter values of the model in the ```models``` directory and a classification report text file in ```out```.

### Results
The logistic model and the neural net perform equally well, given the parameters set for the vectorizer and the neural net model as stated in ``models/vect1.txt`` and ``models/nn_split0.2_vect1.txt``, respectively.

|Model|Overall accuracy|
|---|---|
|log|0.76|
|nn|0.77|

### Setup
The scripts require the following to be run from the terminal:

```shell
bash setup.sh
```

This will create a virtual environment, ```assignment2_env```, to which the packages listed in ```requirements.txt``` will be downloaded. __Note__, ```setup.sh``` works only on computers running POSIX. Remember to activate the environment running the following line in a terminal before changing the working directory to ``src`` and running the ```.py```-scripts.

```shell 
source ./assignment2_env/bin/activate
```
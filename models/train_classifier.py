import sys
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier



def load_data(database_filepath):
    '''
    Load the data from sqlite database.
    
   
    Args:
    database_filepath: string. path to sqlite database
    
    Returns:
    X: pd.Series . data consist of message string data for the X variable .
    Y: pd.DataFrame . data consist of classification of the message types.
    category_names = pd.Series .  data consist of category name
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    category_names=Y.columns
    return  X, Y, category_names


def tokenize(text):
    '''
    Tokenize the message 
   
    Args:
    text: string. message to be tokenize
    
    Returns:
    clean_tokens: list . list of tokenize word in the messages
    '''
    
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build model for machine learning pipeline
   
    Args:
    None
    
    Returns:
    model: Model to run the machine learning
    '''
    
    pipeline = Pipeline([
    
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': [(1,1),(1, 2)],
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf':  [False],
        'clf__estimator__criterion':['gini','entropy'],
        'clf__estimator__min_samples_split': [2,3]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model and print the evaluation statistic for each category name
   
    Args:
    model: trained machine learning model
    X_test: pd.Series. test data of messages
    Y_test: pd.DataFrame. test data for classification of category
    category_names: category name for the Y_test data
    
    Returns:
    None
    '''
    
    
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred,columns=category_names,index=Y_test.index)
 
    for column in category_names:
        print(column)
        print(classification_report(Y_test[column],Y_pred_df[column]))


def save_model(model, model_filepath):
    '''
    Save the model to pickle file
   
    Args:
    model: trained machine learning model
    model_filepath: string. model_path to save in pickle file
    
    Returns:
    None
    '''
    
    pickle.dump(model,open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
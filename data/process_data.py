import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data messages and categories from a specific .csv

    Args:
    messages_filepath: string. Consist of directory to message csv file path
    categories_filepath: string. Consist of directory to categories csv file path

    Returns:
    df: DataFrame. dataframe of merge two dataset
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on ='id')
    return df


def clean_data(df):
    '''
    Clean dataframe to transform category column to a split category column with value 0 or 1.
    Clean dataframe to be a unique value in dataset.

    Args:
    df:DataFrame. dataframe from previous load_data function

    Returns:
    df:DataFrame. cleaned data set
    '''
    # create a dataframe of the 36 individual category columns
    split_category = df['categories'].str.split(';',expand=True)
    split_category.head()
    
    # select the first row of the categories dataframe
    row = split_category.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x : x[:-2])
    
    # rename the columns of `categories`
    split_category.columns = category_colnames
    
    
    for column in split_category:
        # set each value to be the last character of the string
        split_category[column] = split_category[column].str[-1]

        # convert column from string to numeric
        split_category[column] = split_category[column].astype(int)

    
    df.drop(columns='categories',inplace=True)
    df = pd.concat([df,split_category],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    Save the cleaned data to sqlite database.
    
   
    Args:
    df: DataFrame. dataframe from previous load_data function
    database_filename: string. name for the database.
    
    Returns:
    None
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
#### loading libraries
import pandas as pd
import math
import re
import heapq
import regex
import os
from collections import defaultdict, Counter
import nltk
from nltk.stem import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import json
import requests
from tqdm import tqdm
import numpy as np
import heapq
from fuzzywuzzy import fuzz


def search_engine_full(term_ids, df, inverted_index):
    appearances = list()
    [appearances.append(inverted_index[str(term)]) for term in term_ids]
# initialising the set with the value of the first list of appearances in the list of appearances of the word
    intersection_list = set(appearances[0])
    for appearance in appearances[1:]: # for the rest of the terms in the list of term ids
        intersection_list.intersection_update(appearance) # intersect and update the set
    rows_to_append = [] # will be used to create the dataframe
    for row in intersection_list:
        new_row = {
        'courseName': df.loc[row, 'courseName'], # original courseName column and not the clean ones 
        'universityName': df.loc[row, 'universityName'], # similar argument
        'facultyName': df.loc[row, 'facultyName'],
        'city': df.loc[row, 'city'],
        'country': df.loc[row, 'country'],
        'description': df.loc[row, 'description'],
        'FEE_EUR': df.loc[row, 'FEE_EUR'],
        'url': df.loc[row, 'url'],
        }
        rows_to_append.append(new_row)
    new_df = pd.DataFrame(rows_to_append) # creating the new dataframe to be returned on execution of the function
    return new_df


def verify_occurences(df, column, query):
    """
    Args:
        df (DatFrame): dataframe
        column (str): name of the column
        query (str): input query

    Returns:
        Bool: True if an element of query matches a cell of the df[column]
    """
    for elem in query.lower().split():
        for elem_b in list(map(str.lower, df[column].tolist())):
            if elem in elem_b:
                return True
    return False


def calculate_weights(query, df):
    """computes the values of the weights verifying if at least a 
    word in the query matches the column of the df

    Args:
        query (str): input query
        df (DataFrame): dataframe

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if df.empty:
        raise ValueError
    # checking for the most interesting columns if a word of the query matches
    # the contents of the dataframe columns
    value_courseName = 5 if verify_occurences(df, 'courseName', query) else 1
    value_description = 2 if verify_occurences(df, 'description', query) else 0.5
    value_universityName = 5 if verify_occurences(df, 'universityName', query) else 1
    value_city = 5 if verify_occurences(df, 'city', query) else 1
    value_country = 5 if verify_occurences(df, 'country', query) else 1
    value_faculty = 5 if verify_occurences(df, 'facultyName', query) else 1
    total = value_courseName + value_description + value_universityName + value_city + value_country + value_faculty
    # Normalizing the weights
    weights = {
        'description': value_description/total,
        'courseName': value_courseName/total,
        'universityName': value_universityName/total,
        'city': value_city/total,
        'country': value_country/total,
        'facultyName': value_faculty/total
    }
    return weights

def calculate_total_score(query, df, weights):
    """calculus of the new score function based on different columns,
    and on the Levenshtein Distance.

    Args:
        query (str): input query
        df (DataFrame): dataframe
        weights (dict): dictionary containing the weights of the columns of the df

    Returns:
        float: value of the new_score
    """
    total_score = 0
    
    for variable, weight in weights.items():
        if variable in df and isinstance(df[variable], str):
            # Calculate text similarity based on the ratio (using fuzzywuzzy library)
            similarity_score = fuzz.ratio(query, df[variable])
            # Normalize the similarity score between 0 and 1
            normalized_similarity = similarity_score / 100.0
            # Calculate the weighted score for the current variable
            variable_score = normalized_similarity * weight
            total_score += variable_score

    return total_score

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

def normalize_query(query):
    query = query.split(" ")
    good_words = [] 
    for word in query:
        word = tokenizer.tokenize(word)
        for theword in word:
            theword.lower()
            if theword.isalpha() and theword not in stopwords_list:
                theword = stemmer.stem(theword)
                good_words.append(theword)
    return good_words

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


### BONUS QUESTIONS 
# 5.1

def search_1 (normalized_query_1,normalized_query_2,normalized_query_3, df):
    normalized_query_1 = normalize_query(user_input_1)
    normalized_query_2 = normalize_query(user_input_2)
    normalized_query_3 = normalize_query(user_input_3)

    term_ids_1 = get_term_id_from_query(vocab_with_index_1, normalized_query_1) 
    appearances = list()
    [appearances.append(inverted_index_1[str(term)]) for term in term_ids_1] 
    # initialising the set with the value of the first list of appearances in the list of appearances of the word
    intersection_list_1 = set(appearances[0])
    for appearance in appearances[1:]: # for the rest of the terms in the list of term ids
        intersection_list_1.intersection_update(appearance) 
    intersection_list_1 = sorted(list(intersection_list_1))

    term_ids_2 = get_term_id_from_query(vocab_with_index_2, normalized_query_2) 
    appearances = list()
    [appearances.append(inverted_index_2[str(term)]) for term in term_ids_2] 
    intersection_list_2 = set(appearances[0])
    for appearance in appearances[1:]: # for the rest of the terms in the list of term ids
        intersection_list_2.intersection_update(appearance) 
    intersection_list_2 = sorted(list(intersection_list_2))

    term_ids_3 = get_term_id_from_query(vocab_with_index_3, normalized_query_3) 
    appearances = list()
    [appearances.append(inverted_index_3[str(term)]) for term in term_ids_3] 
    # initialising the set with the value of the first list of appearances in the list of appearances of the word
    intersection_list_3 = set(appearances[0])
    for appearance in appearances[1:]: # for the rest of the terms in the list of term ids
        intersection_list_3.intersection_update(appearance) 
    intersection_list_3 = sorted(list(intersection_list_2))

    query_tfidf_1 = calculate_query_tfidf(normalized_query_1, word_and_appearances_1, df)
    query_tfidf_2 = calculate_query_tfidf(normalized_query_2, word_and_appearances_2, df)
    query_tfidf_3 = calculate_query_tfidf(normalized_query_3, word_and_appearances_3, df)

    similarity_scores_1 = {}
    for document_i in intersection_list_1:
        document_vector = df.at[document_i, "courseName_clean"] # getting the document vector at the row of interest
        # calculated the tf_idf of the document vector. It will result in words as keys and values as tf-idf scores
        document_tfidf = calculate_document_tfidf(document_vector, word_and_appearances_1, df) 
        # Compute cosine similarity
        dot_product = 0
        for word in normalized_query_1:
            if word in document_tfidf:
    # computing the dot product of only the query's tfidf score and the tfidf score of that word in the document
                dot_product += query_tfidf_1[word] * document_tfidf[word] 
    
        norm_doc_i = np.linalg.norm(list(document_tfidf.values())) # computing norm of the doc vector
        norm_query = np.linalg.norm(list(query_tfidf_1.values()) )# computing norm of the query vector
    
        if norm_doc_i != 0 and norm_query != 0: # only the non zero results
            cosine_similarity_doc_i_query = dot_product / (norm_doc_i * norm_query)
            similarity_scores_1[document_i] = cosine_similarity_doc_i_query

    similarity_scores_2 = {}
    for document_i in intersection_list_2:
        document_vector = df.at[document_i, "universityName_clean"] # getting the document vector at the row of interest
        # calculated the tf_idf of the document vector. It will result in words as keys and values as tf-idf scores
        document_tfidf = calculate_document_tfidf(document_vector, word_and_appearances_2, df) 
        # Compute cosine similarity
        dot_product = 0
        for word in normalized_query_2:
            if word in document_tfidf:
    # computing the dot product of only the query's tfidf score and the tfidf score of that word in the document
                dot_product += query_tfidf_2[word] * document_tfidf[word] 
    
        norm_doc_i = np.linalg.norm(list(document_tfidf.values())) # computing norm of the doc vector
        norm_query = np.linalg.norm(list(query_tfidf_2.values()) )# computing norm of the query vector
    
        if norm_doc_i != 0 and norm_query != 0: # only the non zero results
            cosine_similarity_doc_i_query = dot_product / (norm_doc_i * norm_query)
            similarity_scores_2[document_i] = cosine_similarity_doc_i_query

    similarity_scores_3 = {}
    for document_i in intersection_list_3:
        document_vector = df.at[document_i, "city_clean"] # getting the document vector at the row of interest
        # calculated the tf_idf of the document vector. It will result in words as keys and values as tf-idf scores
        document_tfidf = calculate_document_tfidf(document_vector, word_and_appearances_3, df) 
        # Compute cosine similarity
        dot_product = 0
        for word in normalized_query_3:
            if word in document_tfidf:
    # computing the dot product of only the query's tfidf score and the tfidf score of that word in the document
                dot_product += query_tfidf_3[word] * document_tfidf[word] 
    
        norm_doc_i = np.linalg.norm(list(document_tfidf.values())) # computing norm of the doc vector
        norm_query = np.linalg.norm(list(query_tfidf_3.values()) )# computing norm of the query vector
    
        if norm_doc_i != 0 and norm_query != 0: # only the non zero results
            cosine_similarity_doc_i_query = dot_product / (norm_doc_i * norm_query)
            similarity_scores_3[document_i] = cosine_similarity_doc_i_query

# Combine similarity scores from all three sources and calculate mean similarity
    mean_similarity_scores = {}

    for document_index in intersection_list_1:
        similarity_scores = [
            similarity_scores_1.get(document_index, 0),
            similarity_scores_2.get(document_index, 0),
            similarity_scores_3.get(document_index, 0)
        ]

    # Calculate the mean similarity
        mean_similarity = sum(similarity_scores) / len(similarity_scores)
        mean_similarity_scores[document_index] = mean_similarity

# Sort mean similarity scores in descending order
    sorted_mean_scores = sorted(mean_similarity_scores.items(), key=lambda x: x[1], reverse=True)

# Iterate through sorted scores
    for document_index, mean_similarity_score in sorted_mean_scores:
    # Check if mean similarity is greater than 0
        if mean_similarity_score > 0:
        # Retrieve master program name from the DataFrame
            master_program_name = df.at[document_index, "courseName"]
            master_program_uni = df.at[document_index, "universityName"]
            master_program_city = df.at[document_index, "city"]
            master_program_url = df.at[document_index, "url"]

        
        # Print master programs and mean similarity score
            print(f"{master_program_name},{master_program_uni},{master_program_city},{master_program_url}, Similarity: {mean_similarity_score}")


# 5.2

def search_2(min_fee, max_fee, df):
    # Filter the DataFrame based on the user-specified fee range
    fee_df = df[(df['FEE_EUR'] >= min_fee) & (df['FEE_EUR'] <= max_fee)]

    # Display the selected columns for the filtered programs
    result_df = fee_df[['courseName', 'universityName', 'url' , 'FEE_EUR']]
    return (result_df)
    

# 5.3

def search_3(selected_countries, df):
# Normalize the selected countries using the normalize_query function
    normalized_countries = [normalize_query(country) for country in selected_countries]

# Filter the DataFrame based on the normalized selected countries
    filtered_country_df = df[df['country_clean'].isin(normalized_countries)]

# Display the selected columns for the filtered programs in the specified countries
    result_country_df = filtered_country_df[['courseName', 'universityName', 'url']]
    return (result_country_df)
    
    
# 5.5


def search_5 (query_modality, df):
    #Normalize input
    normalized_online = normalize_query(query_modality) 
    filtered_df_modality = df[df['administration_clean'].apply(lambda x: isinstance(x, str) and normalized_online in normalize_query(x))]

# Select and print specific columns: courseName, universityName, and url
    result_df_modality = filtered_df_modality[['courseName', 'universityName', 'url']]
    return (result_df_modality)

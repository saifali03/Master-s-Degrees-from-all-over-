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

#### essential prep for pre-processing text
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stopwords_list = set(stopwords.words('english'))
tokenizer = RegexpTokenizer("[\w']+") # function to recognize the tokens

#### creats and loads dataset from paths (locally)
def create_store_tsv(directory, path_to_store):
    all_files = os.listdir(directory) # getting a list of files within the directory
    dataframes = [] # to store all the tsv's, later to be appended into var dataset
    # Loop through each file in the directory
    for filename in all_files:
        if filename.endswith('.tsv'):  # Check if the file is a TSV file
            file_path = os.path.join(directory, filename) # getting file path
            df = pd.read_csv(file_path, sep='\t')  # Read TSV file using tab delimiter
            dataframes.append(df)
    dataset = pd.concat(dataframes, ignore_index=True) # appending dataframes into dataset
    dataset.to_csv(path_to_store, sep='\t', index=False) # storing locally
    return dataset

#### preprocess all columns
def preprocessing(text):
    if isinstance(text, str): # to avoid error in case entry in NULL/NaN
        words_to_keep = [] # tokenized words that will be used for vocabulary
        words = tokenizer.tokenize(text)
        for word in words:
            word_lower = word.lower()
    # check if word is not in stopwords_list and that it is indeed alpha_numeric (otherwise, don't retrieve)
            if word_lower not in stopwords_list and word_lower.isalpha():
                stemmed_word = stemmer.stem(word_lower) # stem it
                words_to_keep.append(stemmed_word) # append
        return words_to_keep
    return text # if the text is not a string, return the original text (NaN / NULL)

#### will replace any appearance of words: ["website", "webpage", "contact"] with "Not Available"
def if_website_or_contact_replace(df, column_name):
    new_column = "fees_clean" # new column to avoid modification of the original column_name
    mask = df[column_name].str.contains('website|webpage|contact', case=False, na=False)
    df[new_column] = df[column_name].copy() # Creating the new column
    # Setting entries to Not Available where the mask is True
    df.loc[mask, new_column] = "Not Available"

#### Extracting Symbols used throught the fees column: helps design regex accordingly
def extract_symbols(df):
    list_of_symbols = set() # initialisation
    for index, row in df.iterrows():
        text = row["fees_clean"] # getting into the fees_clean column
        # using regex, r'\p{Sc}' will find all occurencies of any currency symbol
        symbol_currencies = regex.findall(r'\p{Sc}', text)
        if symbol_currencies: # if there is a find
            list_of_symbols.update(symbol_currencies) # add to the set
    list_of_symbols = list(list_of_symbols) # convert to list
    return list_of_symbols

####  function that will normalize the each row entry of the fees_clean column
#### will replace the test with the max value in the entry + the currency name
#### return example: 1500.0 EUR
def extract_numeric_and_currency(text):
    if text == "Not Available": # nothing to process text contains "Not Available"
        return "Not Available"

    symbol_currency = None # initialisation
    name_currency = None # initialisation

    # Extracting numeric values
    numerics = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text)
    # got stuck with the regex above badly, still do not know how it works because it is pretty complex. Help: GPT 
    if numerics: # if found
        value_list = [float(value.replace(',', '')) for value in numerics] # replace all , with ''
        max_value = max(value_list) # find the max numerical value as requested in HW3
    else:
        max_value = None

    # Extracting currency symbols; pattern adjusted according to symbols found using functon extract_symbols
    symbol_currencies = re.findall(r'[\$\€\£]', text) 
    # it is possible that the row contains fees in different currencies, like: "trisemester 300$ or 280€"
    # simplification made: only the first find will be currency to be returned 
    symbol_currency = symbol_currencies[0] if symbol_currencies else None

    # Extracting currency names; currency patter made after trial and testing: df.fees.str.contains("euro") for example 
    currency_codes = r'USD|EUR|EURO|EUROS|euro|euros|eur|HKD|GBP|JPY|INR|BTC|RUB|TRY|KRW|THB|UAH|KZT|CHF|CRC|NGN|MNT|PHP|PYG|VND|GHS|AUD|CAD|SGD|NZD|ZWD|BSD|BBD|BZD|BMD|SBD|BND|KYD|FJD|GYD|LRD|SRD|TTD|TWD|XCD'
    pattern = re.compile(currency_codes)
    name_currencies = pattern.findall(text)
    # simplification made: only the first find will be currency to be returned 
    name_currency = name_currencies[0] if name_currencies else None

    # Determine output based on extracted values
    # we want the format to be: value + currency_name
    if max_value is not None and name_currency is not None:
        if name_currency.lower() in {"EURO","euro", "euros", "eur"}:
            output = str(max_value) + " " + "EUR"
        else:
            output = str(max_value) + " " + name_currency.upper()

    elif max_value is not None and symbol_currency is not None:
        if symbol_currency == "$":
            output = str(max_value) + " " + "USD"
        elif symbol_currency == "€":
            output = str(max_value) + " " + "EUR"
        else:
            output = str(max_value) + " " + "GBP"
    elif name_currency is None and symbol_currency is None:
        output = str(max_value) + " " + "EUR" # if there is an entry with no currency symbol/abbre, I assume it to be EUR
    else:
        output = "Not Available" # if no condition met
    return output

#### Following functions Extract numeric value and currency name respectively, to be used for API later
def extract_numeric_value(x):
    if x != 'Not Available':
        parts = x.split(' ')
        if len(parts) == 2:
            try:
                return float(parts[0])
            except ValueError:
                pass  # Allow the function to return None by default

    return None
def extract_currency_name(x):
    if x != 'Not Available':
        parts = x.split(' ')
        if len(parts) == 2:
            return parts[1]
    return None

#### function to standardise the fee column in EUR
def convert_to_eur(amount, currency, data):
    if data["result"] == "success": # if key "result" is "success" (following the structure of the json file)
      if currency in data["conversion_rates"]: # if the currency to be converted is in conversion_rates key of the json file
        return amount / data["conversion_rates"][currency] # do conversion
      else: # if currency not in conversion rates
        return amount # we will keep it as it is
    else: # if no success
      return None

#### function to create inverted_index 
def inverted_index(vocab_with_index, df):
    # vocab_with_index is a dict() with words as keys and values as a unique sequence of integers (term_id starting from 0)
    inverted_ind = {term_id: [] for term_id in vocab_with_index.values()}  # Initialization
    for i, row in df.iterrows():
        doc_id = i # poisiton of the document OR the document id
        if len(df.at[i, "description_clean"]) > 0: # checking if the entry (row) is not empty 
            description_words = df.at[i, "description_clean"] # getting the words in that row.
            for word in description_words:
                if word in vocab_with_index: # if that word is also present in our vocabulary
                    term_id = vocab_with_index[word] # getting the term_id
                    if doc_id not in inverted_ind[term_id]:
                        inverted_ind[term_id].append(doc_id) # appending the doc_id as value to the term_id key of inverted_ind dict()
    return inverted_ind

#### function to normalize_query; follows patterns of the preprocessing function.
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

#### function to extract the term_id of the word(s) in query if the word(s) is in the vocab_with_index dictionary
def get_term_id_from_query(vocab_with_index, normalized_query):
    term_ids = list() # initialisation
    for word in normalized_query:
        if word in vocab_with_index.keys():
            term_ids.append(vocab_with_index[word])
    return term_ids

#### our first search engine. term_ids is a list extracted using get_term_id_from_query function
def search_engine(normalized_query, df, words_and_appeareances):
    appearances = list()
    [appearances.append(words_and_appeareances[word]) for word in normalized_query]
# initialising the set with the value of the first list of appearances in the list of appearances of the word
    intersection_list = set(appearances[0])
    for appearance in appearances[1:]: # for the rest of the terms in the list of term ids
        intersection_list.intersection_update(appearance) # intersect and update the set
    rows_to_append = [] # will be used to create the dataframe
    for row in intersection_list:
        new_row = {
        'courseName': df.loc[row, 'courseName'], # original courseName column and not the clean ones 
        'universityName': df.loc[row, 'universityName'], # similar argument
        'description': df.loc[row, 'description'],
        'url': df.loc[row, 'url'] 
        }
        rows_to_append.append(new_row)
    new_df = pd.DataFrame(rows_to_append) # creating the new dataframe to be returned on execution of the function
    return new_df

def inverted_index_tfidf(word_and_appearances, df):
    """
    Term Frequency: TF of a term or word is the number of times the term appears in a document compared to the total
    number of words in the document. 
    Inverse Document Frequency: Number of documents in the corpus divided by the number of documents in the corpus 
    that contain the term.
    Source learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/
    """
    new_index = {} # a dict with key as term_id of the word and as value: a tuple(row_at_which_word_occured, tfidf)
    words = list(word_and_appearances.keys()) # getting all the words of the vocabulary
    for word in words:
        appearances = word_and_appearances[word] # getting all the places (rows of df) where the word occured
        list_of_tuples = [] # to store list of tuple (row_at_which_word_occured, tfidf)
        for df_position in appearances:
            df_row = df.at[df_position, "description_clean"] # going at that row
            tf = df_row.count(word) / len(df_row) # computing the term frequency
            idf = math.log(len(df)/len(appearances)) # computing the inverse doc frequency
            tf_idf = round(tf * idf, 2) # # computing the tf * idf
            list_of_tuples.append((df_position,tf_idf)) # appending to the list ot tuples
        new_index[words.index(word)] = list_of_tuples # filling the index
    return new_index

# Computes TF-IDF for the query
def calculate_query_tfidf(normalized_query, word_and_appearances, df):
    query_tfidf = {} # a dict with key as the word and as value the tfidf
    for word in normalized_query:
        if word in word_and_appearances:
            appearances = word_and_appearances[word] # similar arguments as the inverted_index_tfidf
            tf = normalized_query.count(word) / len(normalized_query)
            idf = math.log(len(df) / len(appearances))
            tf_idf = round(tf * idf, 2)
            query_tfidf[word] = tf_idf
    return query_tfidf

# Computes TF-IDF for the document
def calculate_document_tfidf(document_vector, word_and_appearances, df):
    document_tfidf = {} # a dict with key as the word and as value the tfidf
    for word in document_vector:
        if word in word_and_appearances:
            appearances = word_and_appearances[word] # similar arguments as the inverted_index_tfidf
            tf = document_vector.count(word) / len(document_vector)
            idf = math.log(len(df) / len(appearances))
            tf_idf = round(tf * idf, 2)
            document_tfidf[word] = tf_idf
    return document_tfidf

##### returns the result (in a dataframe) for the top 10 documents
def return_results(top_10_documents, similarity_scores, df):
    rows_to_append = [] # original dataframe rows that will depend top_10_documents
    for row in top_10_documents:
        new_row = {
        'courseName': df.loc[row, 'courseName'],  # Original courseName column, not the clean one
        'universityName': df.loc[row, 'universityName'],  # Similar argument
        'description': df.loc[row, 'description'],  # Similar argument
        'url': df.loc[row, 'url'],  # Similar argument
        'Cosine_Similarity': similarity_scores[row] # corresponding similarity score
        }
        rows_to_append.append(new_row)
    new_df = pd.DataFrame(rows_to_append)  # Creating the new DataFrame to be returned on execution of the function
    return new_df
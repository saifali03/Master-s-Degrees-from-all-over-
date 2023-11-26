# Algorithmic Methods of Data Mining (Sc.M. in Data Science). Homewok 3

## Team members
* Riccardo Corrente 1964746
* Branescu Ana Carina 2125078
* INSERT NAME SURNAME 

The repository contains the submission of the third homework for the course "Algorithmic Methods of Data Mining", for the Group #22.
## Contents

**NB**: We've identified recent changes on certain pages of the website since the time our web crawler was last run. The results obtained may not be entirely accurate compared to the current content on the website.

* __`main.ipynb`__
    > The Jupyter notebook with the solutions to all the questions. The cells are already executed.
* __`functions`__
    > A folder containing all the .py files used and imported in the `main.ipynb`, in order to improve readability in reporting and efficiency in deploying the code. The folder includes:
    * __`crawler.py`__
        > The bot that browses the website, sending HTTP requests.
    * __`parser.py`__
        > The bot that scrapes informations in the website in order to create the dataframe we are working on.
    * __`engine.py`__
        > Module containing all the functions used to create the Search Engine for Q2.
    * __`functions.py`__
        > Module containing all the functions imported in the `main.ipynb` to solve the questions.
* __`map_courses.html`__
    > The map obtained as the result of Q4, displaying universities filtered through Q3 using the query 'data science'.
* __`CommandLine.sh`__
    > The command line scripts used to answer the questions.
* __`CommandLine_output.png`__
    > The image showing the outputs of the command line scripts.
* __`Q1 outputs`__
    > A folder containing all the outputs of Q1. The folder includes:
    * __`course_links.txt`__
        > The file with the 6000 URLs of the Master Degrees, scraped from the [MSc Degrees Website](https://www.findamasters.com/masters-degrees/msc-degrees/)
    * __`file_tsv.zip`__
        > A zip file containing the 6000 .tsv files for each Master Degree, with the informations scraped.
    * __`unique_tsv_file.tsv`__
        > The .tsv file which merges all the 6000 files. After importing this file in the `main.ipynb` we will create the dataframe we used for the nest questions.
    * __`master_programs_html.7z`__
        > A zip file containing the 6000 HTML pages of the Master Courses, organized in 400 folders with 15 pages each.
* __`Q2 outputs`__
    > A folder containing all the outputs of Q2. The folder includes:
    * __`inverted_index_1.json`__
        > 
    * __`inverted_index_tfidf.json`__
        > 
    * __`vocabulary.json`__
        > 
## Suddivision of exercises
* Q1 - Riccardo
* Q2 - Saif
* Q3 - Riccardo
* Q4 - Riccardo
* Q5 - Ana
* Q6 - Saif
* Q7 - Ana
* Notebook and repository organization - Riccardo
    

# Text Improvement Engine
## Overview
The Text Improvement Engine is a tool designed to analyze a given text and suggest improvements based on the similarity to a list of standardized phrases. The goal of this project is to help users enhance the quality and consistency of their written content by aligning it with predefined standard phrases.

## Features
- User-friendly text analysis and improvement suggestions.
- Utilizes NLTK and scikit-learn for natural language processing and similarity scoring.
- Supports custom lists of standardized phrases.
- Provides similarity scores to gauge the relevance of suggestions.

## Technologies Used
- Python 3.7+
- NLTK (Natural Language Toolkit)
- scikit-learn
- pandas
- CSV file for standard phrases storage

## Ways to run
There you can the code in two ways:
- by the command line which i provide
- in the jupyter notebook or in colab. I have provided the notebook

## Setup
1. **Python Environment:** 

Make sure you have Python 3.7 or a later version installed on your system.

2. **Dependencies:**

Install the required Python libraries using pip:


`pip install nltk scikit-learn pandas`

3. **Standard Phrases:**

Create a CSV file named Standardised terms.csv that contains the standardized phrases you want to use. Each phrase should be in a separate row in the first column.

4. **Input Text:**

Prepare the text you want to analyze and save it in a file named sample_text.txt.

## Usage
Run the provided Python script (text_improvement.py) to analyze the input text and receive improvement suggestions based on the standardized phrases. Make sure to adjust the similarity threshold as needed.


`python text_improve.py 'your input sentence'`

## Design Decisions
**Text Preprocessing:** The input text is preprocessed by tokenizing it into sentences and words, converting to lowercase, and removing stopwords, punctuation, and special characters. This helps improve the accuracy of similarity calculations.

**Cosine Similarity:** Cosine similarity with TF-IDF vectors is used to measure the similarity between input sentences and standardized phrases. It provides a numeric score for similarity.

**Threshold:** A similarity threshold of 0.4 is set to control the suggestions' relevance. You can adjust this threshold based on your specific needs.

**Custom Standard Phrases:** The tool allows you to define custom lists of standardized phrases by modifying the Standardised terms.csv file.

## Example Output
The tool will provide suggestions for improving the input text by replacing phrases with their more "standard" versions, along with similarity scores.

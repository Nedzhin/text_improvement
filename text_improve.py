import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import argparse

# Reading the input text from the command line
parser = argparse.ArgumentParser(description='Text Improvement Engine')
parser.add_argument('input_text', type=str, help='Input text to analyze and improve')
args = parser.parse_args()

# dowloading the tools for text analyzing
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# reading our given standard phrases
standard_phrases = []
with open('Standardised terms.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        standard_phrases.append(row[0])


# When we will read from the file we can use this function

# print(standard_phrases)
# with open('sample_text.txt', 'r') as file:
#     input_text = file.read()
    

# processing function of the text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    cleaned_tokens = [[word.lower() for word in words if word.isalnum() and word.lower() not in stop_words] for words in word_tokens]
    cleaned_sentences = [' '.join(words) for words in cleaned_tokens]
    return cleaned_sentences

input_sentences = preprocess_text(args.input_text)


# initialize model for vectorizing the text and trainin it our given standard phrases
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_standard = tfidf_vectorizer.fit_transform(standard_phrases)
tfidf_matrix_input = tfidf_vectorizer.transform(input_sentences)

# with cosine similarity we will find the most similars
similarities = cosine_similarity(tfidf_matrix_input, tfidf_matrix_standard)

# gice the threshold to decide can we replace the original phrase
threshold = 0.4  


# printing the original phrases and their better standard versions with score of similarity
for i, sentence in enumerate(input_sentences):
    similar_indices = [idx for idx, score in enumerate(similarities[i]) if score > threshold]
    
    if similar_indices:
        print(f"Original Sentence {i + 1}: {sentence}")
        for idx in similar_indices:
            print(f"Suggestion: Replace with '{standard_phrases[idx]}' (Similarity Score: {similarities[i][idx]:.2f})")
        print()


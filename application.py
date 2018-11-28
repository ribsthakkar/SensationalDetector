import tensorflow as tf
from tensorflow.contrib import keras
import pattern.en
from pattern.en.wordlist import PROFANITY
import pickle
import nltk
import string
import collections
import pandas
import numpy as np

# ***** Setup vectorizers for features that don't naturally produce numeric values (POS and syntax)
with open("posvect.pkl", "rb") as pik:
    pos_vectorizer = pickle.load(pik)
    print("pickle preloaded for pos vectorizer")

with open("chunkvect_count.pkl", "rb") as pik:
    chunk_vectorizer = pickle.load(pik)
    print("Pickle preloaded for chunk vectorizer")

# Define functions for feature tagging the text
def pos_tag(title, text):
    """
    Uses NLTK to POS tag the titles and the texts of each text
    :param title:
    :param text:
    :return: (vectorized POS for title, vectorized POS for text)
    """
    text_words = nltk.word_tokenize(text)
    stop = nltk.corpus.stopwords.words("english")
    text_words = list(filter(lambda x: x.lower() not in stop and x.lower() not in string.punctuation, text_words))
    tagged_text = [" ".join(x[1] for x in nltk.pos_tag(text_words))]
    title_words = nltk.word_tokenize(title)
    title_words = list(filter(lambda x: x.lower() not in stop and x.lower() not in string.punctuation, title_words))
    tagged_title = [" ".join(x[1] for x in nltk.pos_tag(title_words))]
    return pos_vectorizer.transform(tagged_title), pos_vectorizer.transform(tagged_text)

def profanity_scan(title, text):
    """
    Uses Pattern to count the number of profane words in the text
    :param title:
    :param text:
    :return: return tuple with frequency of profane words in title and text (# of profane in title, # of profane in text)
    """
    profane_list = set(PROFANITY)
    text_words = nltk.word_tokenize(text)
    text_count = 0
    title_count = 0
    for word in text_words:
        if word.lower() in profane_list:
            text_count += 1
    for word in title.split():
        if word.lower() in profane_list:
            title_count += 1
    return title_count, text_count
def sentiment_scan(title, text):
    """
    Uses Pattern to get the polarity and subjectivity score provided by the pattern sentiment module
    :param title:
    :param text:
    :return: Pair of tuples with polarity and subjectivity scores of title and text ((title polarity, title subjectivity), (text polarity, text subjectivity))
    """

    return (pattern.en.sentiment(title), pattern.en.sentiment(text))
def sent_len(title, text):
    """
    Length of title and text
    :param title:
    :param text:
    :return: (length of title, average length of sentences in text)
    """
    total = 0
    text_sent = nltk.sent_tokenize(text)
    for sent in text_sent:
        total += len(nltk.word_tokenize(sent))
    return (len(nltk.word_tokenize(title)), total / len(text_sent))

def syntax(text):
    """
    Use the pattern sentence tree parsing tool to split up the sentence into its chunk permutation
    :param title:
    :param text:
    :return: (chunk permutations of each type of the entire text)
    """
    s = pattern.en.parsetree(text, relations = True, lemmata = True)
    text_chunks = []
    for sentence in s:
        out = ""
        for chunk in sentence.chunks:
            out += str(chunk.type)
        text_chunks.append(out)
    text_chunks_out = [" ".join(text_chunks)]
    return (chunk_vectorizer.transform(text_chunks_out),)

def emphasis(title, text):
    """
    Determine how many uses of exclamations or questions
    :param title:
    :param text:
    :return: (Punctuation in title, punctuation in text)
    """
    title_counts = collections.Counter(title)
    text_counts = collections.Counter(text)
    text_count = 0
    title_count = 0
    exclamatory = ('?', '!')
    for k in exclamatory:
        if title_counts[k] is not None:
            title_count += title_counts[k]
        if text_counts[k] is not None:
            text_count += text_counts[k]
    return text_count, title_count
def capitals(title, text):
    """
    Scan through the text and deteremine how many words are all capital letters
    :param title:
    :param text:
    :return: (# of words in all CAPS of title, # of words in call CAPS of text)
    """
    text_words = nltk.word_tokenize(text)
    text_count = 0
    title_count = 0
    for word in text_words:
        if word.isupper():
            text_count += 1
    for word in title.split():
        if word.isupper():
            title_count += 1
    return title_count, text_count


try:
    print("Loading model")
    saved_model = keras.models.load_model('sensational_detector_model.h5')
except FileNotFoundError:
    print("Model does not exist")
    exit(1)

data = pandas.read_csv("CustomData.csv")
major_errors = 0
journal_errors = 0
error_sources = []
for index, row in data.iterrows():
    title = row['Title']
    text = row['Text']
    expected = row['Sensationalized']
    source = row['Source']

    pos = pos_tag(title, text)
    prof = profanity_scan(title, text)
    sent = sentiment_scan(title, text)
    length = sent_len(title, text)
    synt = syntax(text)
    emp = emphasis(title, text)
    capt = capitals(title, text)

    pos_title = pos[0].todense()[0].tolist()[0]
    pos_text = pos[1].todense()[0].tolist()[0]
    prof_title = [prof[0]]
    prof_text = [prof[1]]
    sent_title = [sent[0][0], sent[0][1]]
    sent_text = [sent[1][0], sent[1][1]]
    len_title = [length[0]]
    len_text = [length[1]]
    synt_text = synt[0].todense().tolist()[0]
    emp_title = [emp[0]]
    emp_text = [emp[1]]
    cap_title = [capt[0]]
    cap_text = [capt[1]]

    output_vector = [pos_title, pos_text, prof_title, prof_text, sent_text, sent_title, len_title, len_text,
                     synt_text, emp_title, emp_text, cap_title, cap_text]

    vect = [keras.preprocessing.sequence.pad_sequences(output_vector,
                                                      value=0,
                                                      padding='post',
                                                      maxlen=128)]

    sample = np.array(vect)

    result = saved_model.predict(sample)
    if abs(result[0] - expected) > .5:
        major_errors += 1
        print(f"The following article, {title} ,was predicted to be {result[0]} with it actually being {expected}")
        error_sources.append(source)
        if source == 'Scientific Journal':
            journal_errors += 1

print(f"Model had {major_errors} major errors")
print(f"Model had {journal_errors} errors from journal articles")
print(f"Error sources are {collections.Counter(error_sources)}")
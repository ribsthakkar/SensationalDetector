import pandas
import collections
import nltk
import string
import pattern.en
from pattern.en.wordlist import PROFANITY


data = pandas.read_csv("training_data.csv")
print("Number of Rows in Sensationalization dataset: ", len(data))
print("Number of Cols in Sensationalization dataset: ", len(data.columns))
c = collections.Counter((row['Source'], row['Sensationalized']) for index, row in data.iterrows())
c2 = collections.Counter(row['Sensationalized'] for index, row in data.iterrows())
for index, row in data.iterrows():
    if index > 10:
        break
    c.update(row['Source'])
    print(row['Source'])

# **** Feature tagging the Data
'''
1. Part of Speech (Title/text)
2. Profanity Scan (Title/text)
3. Sentiment Analysis of text (by sentence of each text)
4. Sentence Length
5. Type of Sentence qualifier (punctuation or interrogative or etc)
6. Sentence Syntax
7. Number of words with all CAPS

'''
profane_list = set(PROFANITY)
all_chunks = set()
def pos_tag(title, text):
    """
    Uses NLTK to POS tag the titles and the texts of each text
    :param title:
    :param text:
    :return: return tuple containing the (POS tagged title, POS tagged text)
    """
    text_words = nltk.word_tokenize(text)
    stop = nltk.corpus.stopwords.words("english")
    text_words = list(filter(lambda x: x.lower() not in stop and x.lower() not in string.punctuation, text_words))
    tagged_text = nltk.pos_tag(text_words)
    title_sent = nltk.sent_tokenize(title)
    title_words = title_sent.split()
    title_words = list(filter(lambda x: x.lower() not in stop and x.lower() not in string.punctuation, title_words))
    tagged_title = nltk.pos_tag(title_words)
    return tagged_title, tagged_text

def profanity_scan(title, text):
    """
    Uses Pattern to count the number of profane words in the text
    :param title:
    :param text:
    :return: return tuple with frequency of profane words in title and text (# of profane in title, # of profane in text)
    """
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
def syntax(title, text):
    """
    Use the pattern sentence tree parsing tool to split up the sentence into its chunk permutation
    :param title:
    :param text:
    :return: (chunk permutation of title, chunk permutations of each sentence in the text)
    """
    s = pattern.en.parsetree(text, relations = True, lemmata = True)
    t = pattern.en.parsetree(title, relations = True, lemmata = True)
    text_chunks = []
    title_chunks = []
    for sentence in s:
        out = ""
        for chunk in sentence.chunks:
            out += str(chunk.type)
        all_chunks.add(out)
        text_chunks.append(out)

    for sentence in t:
        out = ""
        for chunk in sentence.chunks:
            out += str(chunk.type)
        all_chunks.add(out)
        title_chunks.append(out)

    return title_chunks, text_chunks

def type(title, text):
    """
    Determine how many sentences end in exclamations or questions
    :param title:
    :param text:
    :return: (Punctuation in title, punctuation in text)
    """
    title_counts = collections.Counter(title)
    text_counts = collections.Counter(text)
    text_count = 0
    title_count = 0
    for k in string.punctuation:
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



### Iterate through each of the text, title combinations in the data. Compute the features and write to a pandas frame

### Store the pandas frame in another CSV


### Go through the feature tagged data and all of the features in the model (particularly the chunk permutations
#   and respective punctuation count of the text)
# output all of these features for each title and text combination in purely numerical data format

# **** Training the neural model for the data

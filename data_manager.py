import pandas
import collections
import nltk
import string
import pattern.en
from nltk.data import  load
from pattern.en.wordlist import PROFANITY
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import numpy as np
import re
import tensorflow as tf
from tensorflow.contrib import keras

computed = False
try:
    data = pandas.read_csv("processed_data3.csv")
    computed = True
    print("finished reading processed")
except:
    print("processed had an error")
    data = pandas.read_csv("training_data.csv")

print("Number of Rows in Sensationalization dataset: ", len(data))
print("Number of Cols in Sensationalization dataset: ", len(data.columns))
# c = collections.Counter((row['Source'], row['Sensationalized']) for index, row in data.iterrows())
# c2 = collections.Counter(row['Sensationalized'] for index, row in data.iterrows())
# for index, row in data.iterrows():
#     if index > 10:
#         break
#     c.update(row['Source'])
#     print(row['Source'])

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

#### Setup vectorizers for features that don't naturally produce numeric values (POS and syntax)
try:
    with open("posvect.pkl", "rb") as pik:
        pos_vectorizer = pickle.load(pik)
        print("pickle preloaded for pos vectorizer")
except:
    pos_tags = load('help/tagsets/upenn_tagset.pickle').keys()
    pos_vectorizer = CountVectorizer(token_pattern='[^\s]+')
    pos_vectorizer.fit(pos_tags)
    print(pos_tags)
    print(pos_vectorizer.vocabulary_)
    pickle.dump(pos_vectorizer, open("posvect.pkl", "wb"))
    print("pickle created for pos vectorizer")

try:
    with open("chunkvect_count.pkl", "rb") as pik:
        chunk_vectorizer = pickle.load(pik)
        print("Pickle preloaded for chunk vectorizer")
except:
    popular_unsens = {('NP', 4804), ('NPVPNP', 4405), ('NPVPNPPPNP', 2385), ('NPNP', 2378), ('NPVPPPNP', 2308), ('', 2251),
     ('NPVP', 1920), ('VPNP', 1589), ('NPVPNPVPNP', 1222), ('NPVPNPNP', 1213), ('NPPPNP', 1185), ('NPVPPPNPPPNP', 1012),
     ('NPVPNPPPNPPPNP', 1000), ('NPNPVPNP', 900), ('NPVPNPVPNPPPNP', 725), ('PPNP', 669), ('NPVPNPVPPPNP', 653),
     ('NPVPADJP', 571), ('NPVPNPVP', 533), ('NPNPPPNP', 524), ('NPNPVPNPPPNP', 514), ('VPNPPPNP', 493), ('NPNPNP', 471),
     ('ADVPNP', 449), ('VPNPNPVP', 433), ('NPVPPPNPVPNP', 433), ('NPVPNPPPNPNP', 431), ('NPPPNPPPNP', 418),
     ('NPPPNPVPNP', 409), ('NPVPPPNPPPNPPPNP', 402), ('NPNPVPPPNP', 402), ('NPVPNPNPVP', 400), ('NPVPNPPPNPVPNP', 400),
     ('VPNPVPNP', 367), ('NPADVP', 356), ('NPVPNPNPPPNP', 340), ('NPPPNPNP', 332), ('NPVPNPPPNPPPNPPPNP', 326),
     ('NPVPPPNPNP', 316), ('NPVPNPPPNPVPNPPPNP', 311), ('NPVPADVP', 309), ('VPPPNP', 308), ('NPVPNPVPNPVPNP', 298),
     ('NPVPNPNPVPNP', 297), ('VPPPVPPPNPPPNPVP', 291), ('NPVPNPADVP', 291), ('PPNPNP', 281),
     ('NPPPNPPPADJPNPNPPPNP', 278), ('NPNPVP', 261), ('VPNPVP', 258), ('NPVPNPPPNPVPPPNP', 253), ('NPVPADJPPPNP', 250),
     ('NPPPNPVPPPNP', 249), ('NPVPPPNPVPPPNP', 247), ('NPVPPPNPVPNPPPNP', 233), ('NPVPPPNPPPNPNP', 226), ('ADVP', 225),
     ('NPNPVPPPNPPPNP', 219), ('NPVPADVPPPNP', 217), ('NPPPNPPPNPNPPPNP', 215), ('NPPPNPVPNPPPNP', 212),
     ('NPVPNPVPPPNPPPNP', 212), ('NPVPNPVPNPVP', 211), ('NPVPNPVPNPPPNPPPNP', 210), ('VPNPNPADJPPPNPPPNP', 209),
     ('NPVPNPVPNPNP', 205), ('NPVPNPPPNPVP', 202), ('VPNPVPNPPPVP', 199), ('PP', 199), ('NPPPNPVP', 196),
     ('NPPPNPNPPPNPNPNP', 194), ('VP', 192), ('VPNPPPNPPPNP', 190), ('NPVPNPVPVP', 188), ('NPNPVPNPPPNPPPNP', 187),
     ('NPPPNPVPADJP', 182), ('NPVPNPPP', 180), ('PPNPNPVPNP', 179), ('NPVPPP', 177), ('VPNPVPPPNP', 175),
     ('VPNPNP', 175), ('NPVPNPPPNPPPNPNP', 174), ('NPVPNPVPADJP', 173), ('VPNPNPADVPPPNPPPNP', 173),
     ('NPNPVPNPNP', 173), ('NPNPNPPPNP', 167), ('NPVPNPNPNP', 165), ('NPVPPPNPPPNPPPNPPPNP', 165),
     ('NPVPNPPPVPNP', 165), ('NPVPADVPNP', 162), ('NPVPNPPPNPNPVP', 162), ('NPVPPPNPVP', 160), ('PPNPVPNP', 156),
     ('NPVPNPVPNPVPPPNP', 156), ('NPVPNPPPNPNPPPNP', 155), ('NPNPVPNPVPNP', 153), ('VPNPVPNPPPNP', 151),
     ('NPVPNPVPNPPPNPNP', 150), ('NPVPNPNPVPPPNP', 149), ('NPNPNPVPNP', 149)}
    popular_sens = {('NP', 3580), ('NPVPNP', 3360), ('NPVPNPPPNP', 1854), ('VPNP', 1842), ('NPVP', 1473), ('NPVPPPNP', 1380),
     ('NPNP', 1102), ('NPVPNPVPNP', 1094), ('', 1088), ('NPPPNP', 874), ('NPVPADJP', 699), ('NPVPNPNP', 685),
     ('NPVPNPPPNPPPNP', 673), ('VPNPPPNP', 670), ('NPNPVPNP', 609), ('NPVPPPNPPPNP', 585), ('NPVPNPVPPPNP', 561),
     ('VPNPVPNP', 553), ('NPVPNPVP', 545), ('NPVPNPVPNPPPNP', 503), ('VPNPVP', 499), ('ADVP', 430),
     ('NPVPNPNPPPNP', 424), ('NPVPPPNPVPNP', 396), ('VP', 391), ('NPVPADVP', 386), ('VPPPNP', 375), ('ADJP', 363),
     ('NPPPNPVPNP', 358), ('NPNPVPNPPPNP', 353), ('NPNPVP', 352), ('NPVPNPPPNPVPNP', 349), ('VPNPVPPPNP', 336),
     ('NPVPNPADVP', 335), ('NPNPPPNP', 322), ('NPNPNP', 319), ('VPNPNP', 307), ('NPVPNPNPVPNP', 306), ('PPNP', 305),
     ('NPVPNPPPNPNP', 301), ('NPNPNPNP', 287), ('NPNPVPPPNP', 279), ('ADVPNPVPNP', 272), ('NPVPNPVPNPVPNP', 271),
     ('NPVPPPNPVPNPPPNP', 270), ('NPPPNPNP', 257), ('NPVPNPNPVP', 252), ('VPNPPPNPPPNPPPNP', 250),
     ('ADVPNPVPNPADVP', 248), ('NPVPADJPPPNP', 247), ('NPPPNPPPNP', 241), ('VPPPNPPPNP', 232), ('VPADJP', 230),
     ('ADVPNP', 224), ('VPNPPPNPPPNP', 219), ('PPNPVPNP', 209), ('NPVPPPNPNP', 208), ('NPVPPPNPVPPPNP', 207),
     ('NPVPNPPPNPVPNPPPNP', 197), ('VPNPVPPP', 197), ('VPNPVPNPPPNP', 187), ('NPPPNPNPVPNP', 186),
     ('NPVPNPVPNPNP', 185), ('NPPPNPVPNPPPNP', 184), ('VPADVP', 184), ('NPVPADVPPPNP', 183), ('NPNPVPNPVPNP', 179),
     ('NPVPNPPPNPPPNPPPNP', 177), ('VPNPPPVPPPNPVPNPNPVPPPNP', 175), ('NPVPPP', 174), ('NPVPNPVPNPVP', 173),
     ('PPNPNPVPNP', 168), ('NPVPNPPPNPVP', 168), ('NPVPPPNPPPNPPPNP', 167), ('PPNPPPNPNPNPVP', 163), ('NPPPNPVP', 161),
     ('NPADJP', 160), ('NPVPPPNPVPADVPPPNPVPADVPVPVPPP', 158), ('NPPPNPVPPPNP', 157), ('NPVPVPNP', 155),
     ('NPADVP', 154), ('NPVPNPNPNP', 153), ('NPVPNPPP', 150), ('VPNPPP', 148), ('NPVPNPVPADJP', 147),
     ('VPNPNPVPNPPPNP', 146), ('NPNPNPVPNP', 146), ('NPVPPPNPVP', 144), ('NPVPNPVPPPNPPPNP', 144), ('PPNPVPNPPP', 144),
     ('ADVPNPVPNPPPNP', 144), ('PPNPVPNPVPNP', 144), ('NPNPVPNPNPVPPPVPPPNPPP', 143), ('NPVPNPPPNPVPPPNP', 143),
     ('NPVPNPVPNPVPPPNPPPNP', 141), ('ADVPNPVP', 141), ('NPVPADVPNP', 140), ('NPVPNPADJP', 139),
     ('NPVPNPNPVPNPPPNP', 136), ('PP', 136)}
    most_popular_chunks = set()
    for pop1, pop2 in zip(popular_sens, popular_unsens):
        most_popular_chunks.add(pop1[0])
        most_popular_chunks.add(pop2[0])
    chunk_vectorizer = CountVectorizer(token_pattern='[^\s]+')
    chunk_vectorizer.fit(most_popular_chunks)
    pickle.dump(chunk_vectorizer, open("chunkvect_count.pkl", "wb"))
    print("Created pickle for the first time for chunk vectorizer")


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

if not computed:
    ### Iterate through each of the text, title combinations in the data. Compute the features and write to a pandas frame
    features = (pos_tag, profanity_scan, sentiment_scan, sent_len, syntax, emphasis, capitals)
    for feature in features:
        print("Running the following feature: ", feature.__name__)
        if feature.__name__ != "syntax" and feature.__name__ != "pos_tag" and feature.__name__ != "sentiment_scan":
            output = (feature(row['Title'], row['Text']) for index, row in data.iterrows())
            title_feats = []
            text_feats = []
            for x in output:
                title_feats.append(x[0])
                text_feats.append(x[1])
            data[feature.__name__ + "_title"] = title_feats
            data[feature.__name__ + "_text"] = text_feats
        elif feature.__name__ == "pos_tag":
            output = (feature(row['Title'], row['Text']) for index, row in data.iterrows())
            title_feats = []
            text_feats = []
            for x in output:
                title_feats.append(x[0].todense()[0])
                text_feats.append(x[1].todense()[0])
            data[feature.__name__ + "_title"] = title_feats
            data[feature.__name__ + "_text"] = text_feats
        elif feature.__name__ == "syntax":
            output = (feature(row['Text']) for index, row in data.iterrows())
            text_feats = []
            for x in output:
                text_feats.append(x[0].todense()[0])
            data[feature.__name__ + "_text"] = text_feats
        else:
            output = (feature(row['Title'], row['Text']) for index, row in data.iterrows())
            title_feats_1 = []
            title_feats_2 = []
            text_feats_1 = []
            text_feats_2 = []
            for x in output:
                title_feats_1.append(x[0][0])
                title_feats_2.append(x[0][1])
                text_feats_1.append(x[1][0])
                text_feats_2.append(x[1][1])
            data[feature.__name__ + "_title_polarity"] = title_feats_1
            data[feature.__name__ + "_title_subjectivity"] = title_feats_2
            data[feature.__name__ + "_text_polarity"] = text_feats_1
            data[feature.__name__ + "_text_subjectivity"] = text_feats_2
    # pickle the data
    data.to_pickle("computed_data2.pkl")

### Store the pandas frame in another CSV
data.to_csv("processed_data2.csv", sep = ";")

def stringToList(string):
    out = re.sub("\s+", ",", string.strip())
    if out[2] == ',':
        out = out.replace(",","", 1)
    return eval(out)

all_data = []
all_labels = []

# **** Training the neural model for the data
for index, row in data.iterrows():
    # print(row['Title'])
    # print(row['Text'])
    # print(stringToList(row['pos_tag_title']))
    # print(stringToList(row['pos_tag_text']))
    # print(row['profanity_scan_title'])
    # print(row['profanity_scan_text'])
    # print(eval(row['sentiment_scan_title']))
    # print(eval(row['sentiment_scan_text']))
    # print(row['sent_len_title'])
    # print(row['sent_len_text'])
    # print(eval(re.sub("\s+", ",", row['syntax_text'].strip())))
    # print(row['emphasis_title'])
    # print(row['capitals_title'])
    # print(row['capitals_text'])
    # print(row['Sensationalized'])
    pos_title = stringToList(row['pos_tag_title'])[0]
    pos_text = stringToList(row['pos_tag_text'])[0]
    prof_title = [int(row['profanity_scan_title'])]
    prof_text = [int(row['profanity_scan_text'])]
    sent_text = eval(row['sentiment_scan_title'])
    sent_title = eval(row['sentiment_scan_text'])
    len_title = [int(row['sent_len_title'])]
    len_text = [float(row['sent_len_text'])]
    syntax_text = stringToList(row['syntax_text'])[0]
    emp_title = [int(row['emphasis_title'])]
    emp_text = [int(row['emphasis_text'])]
    cap_title = [int(row['capitals_title'])]
    cap_text = [int(row['capitals_text'])]
    output_vector = [pos_title, pos_text, prof_title, prof_text, sent_text, sent_title, len_title, len_text,
                     syntax_text, emp_title, emp_text, cap_title, cap_text]
    # all_data.append(output_vector)
    vect = keras.preprocessing.sequence.pad_sequences(output_vector,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=128)
    all_data.append(vect)
    all_labels.append(int(row['Sensationalized']))
    # print(index)

print("Done loading vectors")
model = keras.models.Sequential()
model.add(keras.layers.Conv1D(32, kernel_size=(1,), input_shape=(13, 128)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

all_data = np.array(all_data)
all_labels = np.array(all_labels)

TRAIN_SIZE = 14000
VALIDATION_SIZE = 500

train_data = all_data[:TRAIN_SIZE]
train_labels = all_labels[:TRAIN_SIZE]
test_data = all_data[TRAIN_SIZE:]
test_labels = all_data[TRAIN_SIZE:]

x_val = train_data[:VALIDATION_SIZE]
partial_x_train = train_data[VALIDATION_SIZE:]

y_val = train_labels[:VALIDATION_SIZE]
partial_y_train = train_labels[VALIDATION_SIZE:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

model.summary()
print('----results----')

other = model.evaluate(all_data[TRAIN_SIZE:], all_labels[TRAIN_SIZE:])
print(other)

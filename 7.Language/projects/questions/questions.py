import string

import nltk
import sys
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data = dict()
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        with open(file_path,encoding='utf-8') as f:
            data[file_name] = f.read()

    return data

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    return [word.lower() for word in nltk.tokenize.word_tokenize(document, "english")
            if all([ch not in string.punctuation for ch in word])
            and word not in nltk.corpus.stopwords.words("english")]

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    wordcount = dict()
    wordset = set()
    for document in documents.keys():
        nowDocumentWordSet = set()
        for word in documents[document]:
            if word not in nowDocumentWordSet:
                nowDocumentWordSet.add(word)
                if word in wordset:
                    wordcount[word] += 1
                else:
                    wordcount[word] = 1
                    wordset.add(word)

    return {word: math.log(len(documents.keys()) / wordcount[word])for word in wordset}

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs_dict = dict()
    for document in files.keys():
        tf_idfs = 0

        for queryWord in query:
            tf = sum([1 for word in files[document] if word == queryWord])
            tf_idfs += tf * idfs[queryWord]

        tf_idfs_dict[document] = tf_idfs

    return [i[0] for i in sorted(tf_idfs_dict.items(), key=lambda x: x[1], reverse=True)][:n]



def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    best_sentence = dict()

    for sentence in sentences.keys():
        best_sentence[sentence] = [0, 0]
        for query_word in query:
            if query_word in sentences[sentence]:
                best_sentence[sentence][0] += idfs[query_word]
                best_sentence[sentence][1] += sentences[sentence].count(query) / len(sentences[sentence])

    return [sentence for sentence, value in sorted(best_sentence.items(), key=lambda x: x[1], reverse=True)][:n]

if __name__ == "__main__":
    main()

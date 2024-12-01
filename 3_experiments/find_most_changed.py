#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import gensim
import numpy as np


def calculate_cosine(vector1, vector2):
    """
    Formula for cosine similarity
    """
    cosine_similarity = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cosine_similarity


def file_to_list(txt_file):
    """
    little script to return the first word of each line in a text file
    """
    wordlist = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("\n")[0]
            word = line.split("\t")[0]
            wordlist.append(word)
    return wordlist


def find_most_changed_words(wordlist, m1, m2, top_n=10):
    """
    Inputs:
    -   a list of words to analyse in the form of a txt file
    -   two aligned models
    -   number of words to print
    This function returns the n most changed words from the input word list by measuring the cosine distance between the
    vectors of each model of a word.
    """
    # Load Word2Vec models
    model1 = gensim.models.Word2Vec.load(m1)
    model2 = gensim.models.Word2Vec.load(m2)
    words = file_to_list(wordlist)
    cosine_similarities = {}
    for word in words:
        if word[-5:] != "propn": # to exclude proper nouns

            if word in model1.wv and word in model2.wv:
                vec1 = model1.wv[word]
                vec2 = model2.wv[word]
                cos_sim = calculate_cosine(vec1, vec2)
                cosine_similarities[word] = cos_sim

    # Sort words by cosine similarity, having words with lowest similarity first
    most_changed_words = sorted(cosine_similarities.items(), key=lambda x: x[1])[:top_n]
    print(f"Top {top_n} words that changed the most:")
    for word, dist in most_changed_words:
        print(f"{word}: {dist}")
    for word, dist in most_changed_words:
        print(word)
    return most_changed_words


if __name__ == '__main__':

    lang = "es"
    model1 = f"../models/model_{lang}-new_vec300_win10_mc5_ep5"
    model2 = f"../models/aligned_model_{lang}-old_vec300_win10_mc5_ep5"
    word_list = f'../word_lists/top_words_{lang}_with-pos_only-content.txt'

    find_most_changed_words(word_list, model1, model2)


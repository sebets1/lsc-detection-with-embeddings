#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import gensim
import numpy as np

from scipy.stats import ttest_ind


def calculate_cosine(vector1, vector2):
    """
    Formula for cosine similarity
    """
    cosine_similarity = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cosine_similarity


def file_to_list(txt_file):
    """
    Returns the words at first position per line in a txt file as list of word.
    """
    wordlist = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("\n")[0]
            word = line.split("\t")[0]
            wordlist.append(word)
    return wordlist


def get_cosines_of_list(word_list, model1, model2, exclude_propn=True):
    """
    Calculate all cosine similarities of words in list
    """
    cosine_similarities = []
    for word in word_list:
        if word in model1.wv and word in model2.wv:
            vec1 = model1.wv[word]
            vec2 = model2.wv[word]
            cos_sim = calculate_cosine(vec1, vec2)
            if exclude_propn:
                if word[-5:] != "propn":  # to exclude proper nouns
                    cosine_similarities.append([cos_sim, word])
            else:
                cosine_similarities.append([cos_sim, word])
    return cosine_similarities


def average_change(cosine_list):
    """
    Calculate the mean of the provided cosine scores
    """
    tot_score = 0
    for ele in cosine_list:
        tot_score += ele[0]
    mean = tot_score/len(cosine_list)

    return mean


def t_test(cos_scores1, cos_scores2):
    """
    Perform a t-test to assess if the mean difference is significant.
    """
    group1 = []  # Cosine similarities for group 1
    group2 = []  # Cosine similarities for group 2
    for ele in cos_scores1:
        group1.append(ele[0])
    for ele in cos_scores2:
        group2.append(ele[0])

    # Perform a t-test
    t_stat, p_value = ttest_ind(group1, group2, equal_var=True)  # Use equal_var=False for Welch's t-test

    print(f"\n{80*'-'}\nT-test: t-statistic={t_stat}, p-value={p_value}\n{80*'-'}")


def compare_change_in_en_words(top_words, top_lat_words, m1, m2, top_n=10):
    """
    Inputs:
    -   a list of top en words to analyse in the form of a txt file
    -   a list of all top en words with latin origin (also txt file)
    -   two aligned models
    -   number of words to print
    This function returns the n most and least changed words among English words without Latin origin,
    as well as English words with Latin origin. Also returns their average change.
    """
    model1 = gensim.models.Word2Vec.load(m1)
    model2 = gensim.models.Word2Vec.load(m2)
    other_words = [] # all En top words which are not of Lat origin
    all_words = file_to_list(top_words)
    lat_words = file_to_list(top_lat_words)
    for word in all_words:
        if word not in lat_words:
            other_words.append(word)

    cosine_other_words = get_cosines_of_list(other_words, model1, model2)
    cosine_lat_words = get_cosines_of_list(lat_words, model1, model2)
    # Sort words by cosine similarity, having words with lowest similarity first
    cosine_other_words.sort()
    cosine_lat_words.sort()
    mean_change_other = average_change(cosine_other_words)
    mean_change_lat = average_change(cosine_lat_words)

    # output print
    print(f"Top {top_n} words with Latin origin that changed the most:")
    for ele in cosine_lat_words[:top_n]:
        print(f"{ele[1]}: {ele[0]}")

    print(f"\nTop {top_n} words with Latin origin that changed the least:")
    cosine_lat_words.sort(reverse=True)
    for ele in cosine_lat_words[:top_n]:
        print(f"{ele[1]}: {ele[0]}")
    print(f"\nAverage cosine distance across all words: {mean_change_lat}\n{60*'-'}\n")

    print(f"\nTop {top_n} words without Latin origin that changed the most:")
    for ele in cosine_other_words[:top_n]:
        print(f"{ele[1]}: {ele[0]}")

    print(f"\nTop {top_n} words without Latin origin that changed the least:")
    cosine_other_words.sort(reverse=True)
    for ele in cosine_other_words[:top_n]:
        print(f"{ele[1]}: {ele[0]}")
    print(f"\nAverage cosine distance across all words: {mean_change_other}\n{60 * '-'}\n")

    t_test(cosine_lat_words, cosine_other_words)


if __name__ == '__main__':

    model1 = f"../models/model_en-new_vec300_win10_mc5_ep5"
    model2 = f"../models/aligned_model_en-old_vec300_win10_mc5_ep5"
    word_list_all_top = f'../word_lists/top_words_en_with-pos_only-content.txt'
    word_list_lat_top = f'../word_lists/en_top-words_with_latin_roots_only-content_manual-filtered.txt'

    compare_change_in_en_words(top_words=word_list_all_top, top_lat_words=word_list_lat_top,
                               m1=model1, m2=model2, top_n=10)










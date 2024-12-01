#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import sys
import gensim
import numpy as np


def txt_to_list(txt_file):
    """
    Opens a text file with one word per line and converts it to a python list.
    """
    word_list=[]
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            target_word = line.replace("\n", "")
            word_list.append(target_word)
    return word_list


def txt_to_list_cognates(txt_file, lang):
    """
    Opens the cognate list file and returns a list of words taking the word from the determined column on each line
    in the text file.
    """
    # the text file with cognates is separated into four columns: en, lat, fr, es
    if lang == "en":
        lang_ind = 0
    if lang == "lat":
        lang_ind = 1
    if lang == "fr":
        lang_ind = 2
    if lang == "es":
        lang_ind = 3

    word_list=[]
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            target_word = line.split("\t")[lang_ind]
            target_word = target_word.replace("\n", "")
            word_list.append(target_word)

    return word_list


def find_cosine_similarity(m1, m2, word_list, print_=True):
    """
    Give paths to two aligned embedding models, a list of words as input.
    The function gets the embeddings for each word from the two models, calculates the cosine distance
    and returns the scores in a list. Optionally, also print results.
    """
    cosine_list = []
    scores_list = []
    model1 = gensim.models.Word2Vec.load(m1)
    model2 = gensim.models.Word2Vec.load(m2)
    for word in word_list:
        vector1 = model1.wv[word]
        vector2 = model2.wv[word]
        cosine_similarity = np.dot(vector1, vector2) / (
                np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cosine_list.append([cosine_similarity, word])
        scores_list.append(cosine_similarity)

    if print_:
        cos_list_copy=cosine_list[:]
        cos_list_copy.sort()
        top_list = 30
        index = 0
        for entry in cos_list_copy:
            if index < top_list:
                score = round(float(entry[0]), 2)
                print(f"Cosine Similarity between '{entry[1]}' embeddings: {score}")
                index+=1

    return cosine_list


def compare_cosine_in_cognate_list(input_words, output_file="print"):
    """
    Calculate cosine simalirites for all words in the short cognate list and print out to terminal or save into
    output file.
    """

    lang_list = ["en", "fr", "es"]
    output_list = []
    for lang in lang_list:
        embedding_model_base = f"../models/model_{lang}-new_vec300_win10_mc5_ep5"
        embedding_model_aligned = f"../models/aligned_model_{lang}-old_vec300_win10_mc5_ep5"
        target_words = txt_to_list_cognates(input_words, lang)
        cosine_list = find_cosine_similarity(embedding_model_base, embedding_model_aligned, target_words, print_=False)
        output_list.append(cosine_list)
        cosine_list.sort()
        print(f"\n{lang}\n")
        for ele in cosine_list:
            print(f"{ele[1]}\t{ele[0]}")


    lat_words = txt_to_list_cognates(input_words, "lat")
    output_sorted_by_word = []
    output_sorted_by_word_rounded = []
    output_sorted_by_word.append(f"Latin word\tEnglish word\tEnglish score\tFrench word\tFrench score\t"
                                 f"Spanish word\tSpanish score")
    for i, word in enumerate(lat_words):
        en_word = output_list[0][i][1]
        en_score = output_list[0][i][0]
        fr_word = output_list[1][i][1]
        fr_score = output_list[1][i][0]
        es_word = output_list[2][i][1]
        es_score = output_list[2][i][0]
        output_sorted_by_word.append(f"{word}\t{en_word}\t{en_score}\t{fr_word}\t{fr_score}\t{es_word}\t{es_score}\t\n")
        output_sorted_by_word_rounded.append(f"{word}\t{en_word}\t{round(float(en_score), 3)}\t"
                                             f"{fr_word}\t{round(float(fr_score), 3)}\t{es_word}\t"
                                             f"{round(float(es_score), 3)}\t")

    if output_file=="print":
        for line in output_sorted_by_word_rounded:
            print(line)
    else:
        with open(output_file, 'w', encoding='utf-8') as file:
            for line in output_sorted_by_word:
                file.write(line)


def calc_cosine_distance_of_top_words(input_words, m1, m2, output_file="print"):
    """
    Input: Text file containing a list of words, two aligned models, optionally an output file.

    Calculate cosine distance between vectors of the models for each word and print/save to output file.
    """
    wordlist = txt_to_list(input_words)
    if output_file == "print":
        find_cosine_similarity(m1, m2, wordlist)
    else:
        cosine_scores = find_cosine_similarity(m1, m2, wordlist)
        cosine_scores.sort()
        with open(output_file, 'w', encoding='utf-8') as file:
            for ele in cosine_scores:
                file.write(f"{ele[1]}\t{ele[0]}")
                file.write("\n")


if __name__ == '__main__':

    cognate_list = True
    top_list = False

    if cognate_list:
        input_words = '../word_lists/cognate-list_en-fr-es_top-words.txt'
        target_words = txt_to_list(input_words)
        output_file = f"../results/cognate_list_all_lang_cosine_scores_vec300_win10_mc5_ep5.txt"
        compare_cosine_in_cognate_list(input_words, output_file)
        compare_cosine_in_cognate_list(input_words, "print")

    if top_list:
        lang = "en"
        embedding_model_base = f"../models/model_{lang}-new_vec300_win10_mc5_ep5"
        embedding_model_aligned = f"../models/aligned_model_{lang}-old_vec300_win10_mc5_ep5"
        input_words = f'../word_lists/top_words_{lang}_with-pos_only-content.txt'
        output_file = f"../results/top_words_{lang}_cosine_distance.txt"
        calc_cosine_distance_of_top_words(input_words, embedding_model_base, embedding_model_aligned, output_file)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

from scipy.stats import spearmanr


def spearman_calc(list1, list2):
    """
    Calculates Spearman's correlation and the p-value on two input lists.
    Called from another function.
    """
    corr, pval = spearmanr(list1, list2)
    return corr, pval


def txt_to_list_cognates(txt_file):
    """
    Opens the cognate list file and returns a lists of cosine scores per language.
    Called from another function.
    """
    en_scores = []
    fr_scores = []
    es_scores = []
    with open(txt_file, 'r', encoding='utf-8') as file:
        first_line = True
        for line in file:
            if first_line:
                first_line = False
            else:
                line_list = line.split("\t")
                en_score = line_list[2]
                en_scores.append(float(en_score))
                fr_score = line_list[4]
                fr_scores.append(float(fr_score))
                es_score = line_list[6]
                es_scores.append(float(es_score))

    return en_scores, fr_scores, es_scores


def txt_to_list(txt_file):
    """
    Opens a text file with one word per line and converts it to a python list.
    Called from another function.
    """
    word_list=[]
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            target_word = line.replace("\n", "")
            word_list.append(target_word)
    return word_list


def print_scores(list_):
    cos_list_copy = list_[:]
    cos_list_copy.sort()

    for entry in cos_list_copy:
        score = round(float(entry[0]), 2)
        print(f"Cosine Similarity between '{entry[1]}' embeddings: {score}")


#######################################################################
# evaluation of cognates
#######################################################################
def cognates_spearman_evaluation(file):
    """
    Evaluate the scores from the cognate list. Use spearman evaluation to compare two languages.
    """
    en_scores, fr_scores, es_scores = txt_to_list_cognates(file)

    corr, pval = spearman_calc(en_scores, fr_scores)
    print("Spearman's correlation coefficient of languages en and fr:", corr)
    print("p-value:", pval)
    print("\n")

    corr, pval = spearman_calc(en_scores, es_scores)
    print("Spearman's correlation coefficient of languages en and es:", corr)
    print("p-value:", pval)
    print("\n")

    corr, pval = spearman_calc(fr_scores, es_scores)
    print("Spearman's correlation coefficient of languages fr and es:", corr)
    print("p-value:", pval)
    print("\n")


if __name__ == '__main__':
    cognates_spearman_evaluation(f"../word_lists/cognate_list_all_lang_cosine_scores_vec300_win10_mc5_ep5.txt")




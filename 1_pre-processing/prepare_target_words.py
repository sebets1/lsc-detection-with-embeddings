#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings


import json
import os
import random
from nltk.corpus import stopwords
nltk.download('stopwords')


def save_word_list_to_txt_file(word_list, output_file):
    """
    Saves the top n words (output of the function count_word_occurrences) to a text file.
    """
    with open(output_file, "w", encoding='utf-8') as file:
        for line in word_list:
            file.write(line)
            file.write("\n")


def only_content_words(token):
    content_tags = ['adj', 'adv', 'noun', 'propn', 'verb']
    pos = token.split("_")[-1]

    if pos in content_tags:
        return True
    else:
        return False


def count_word_occurrences(filename, top_n=500, filter_pos=False, remove_stop=True):
    """
    Take a tokenized and lemmatized corpus (as list of sentences) as input.
    Then count lemmas and return a ranked frequency list of the n most common words.
    """
    stop_words = set(stopwords.words('english'))
    word_collection = {}
    with open(filename, 'r', encoding='utf-8') as f:
        loaded_corpus = json.load(f)
        for sent in loaded_corpus:
            if remove_stop:
                sent = [word for word in sent if word.split("_")[0] not in stop_words]  # remove stopwords
            for token in sent:
                check_word = True
                if filter_pos:
                    check_word = only_content_words(token)
                if check_word:
                    if token in word_collection:
                        word_collection[token] += 1
                    else:
                        word_collection[token] = 1

    ranked_words = sorted(word_collection.items(), reverse=True, key=lambda item: item[1])
    filtered_ranked_list = []
    for i, tuple_token in enumerate(ranked_words):
        if i<=top_n:
            print(tuple_token)
            filtered_ranked_list.append(tuple_token[0])
    return filtered_ranked_list


def compare_two_files(input_file1, input_file2, output_file="test.txt"):
    """
    Compare the lists of top n words of the old and new corpus (of a language) and save all words which occur in both
    lists in a new list.
    """
    words1 = []
    matching_lines = []
    with open(input_file1, "r", encoding="utf-8") as f1:
        for line in f1:
            words1.append(line)
    with open(input_file2, "r", encoding="utf-8") as f2:
        for line in f2:
            if line in words1:
                matching_lines.append(line)
    with open(output_file, 'w') as file:
        for line in matching_lines:
            file.write(line)
    return matching_lines


def compare_top_words_to_latin_list(cognate_list, es_top_list, fr_top_list, output_file):
    """
    Check if the words in the latin cognates list also are among the top words of the French and Spanish corpora.
    """
    # the text file with cognates is separated into four columns: en, lat, fr, es
    words_es = []
    words_fr = []
    matching_lines = []
    with open(es_top_list, "r", encoding="utf-8") as f1:
        for line in f1:
            word = line.replace("\n", "")
            words_es.append(word)

    with open(fr_top_list, "r", encoding="utf-8") as f2:
        for line in f2:
            word = line.replace("\n", "")
            words_fr.append(word)

    with open(cognate_list, "r", encoding="utf-8") as f:
        for line in f:
            word_es = line.split("\t")[3]
            word_es = word_es.replace("\n", "")
            if word_es in words_es:
                word_fr = line.split("\t")[2]
                word_fr = word_fr.replace("\n", "")
                if word_fr in words_fr:
                    matching_lines.append(line)
    print(len(matching_lines))

    with open(output_file, 'w') as file:
        for line in matching_lines:
            file.write(line)


def file_to_list(txt_file, row=0):
    """
    little script to return the first word of each line in a text file
    """
    wordlist = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("\n")[0]
            word = line.split("\t")[row]
            wordlist.append(word)
    return wordlist


def random_list(input_file, output_file, n_words=100):
    """
    input: list of words
    shuffle words
    output: randomly sampled list, saved in output text file
    """
    wordlist = file_to_list(input_file)
    random.shuffle(wordlist)
    selected_words = wordlist[:n_words]
    save_word_list_to_txt_file(selected_words, output_file)


#################################################
# for evaluation of automatic etymology detection
#################################################
def check_latin_detection():
    """
    To check how well the 'find_etymology.py' script detects latin words.
    False positives and true positives can be assigned automatically via this script.
    False negatives and true negatives have to be determined manually afterwards.
    """
    random_selected = file_to_list("../word_lists/random_top_words_en_latin-detection-check.txt")
    true_positives = file_to_list("../word_lists/en_top-words_with_latin_roots_only-content_manual-filtered.txt")
    all_positives = file_to_list("../word_lists/en_top-words_with_latin_roots_only-content.txt")
    output_file = "../word_lists/random_top_words_en_latin-detection-check_pre.txt"
    false_positives =[]

    final = []
    for word in all_positives:
        if word not in true_positives:
            false_positives.append(word)

    for word in random_selected:
        if word in true_positives:
            final.append(f"{word}\tTP")
        elif word in false_positives:
            final.append(f"{word}\tFP")
        else:
            final.append(f"{word}")
    save_word_list_to_txt_file(final, output_file)


def confusion_matrix(list_of_tags):
    """
    Input: list of tags as TP, FP, TN and FN
    Output: Calculations of precision, recall, accuracy and f-measure.
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for ele in list_of_tags:
        if ele == "TP":
            TP+=1
        if ele == "TN":
            TN+=1
        if ele == "FP":
            FP+=1
        if ele == "FN":
            FN+=1

    # Calculate precision, recall, accuracy, and F-measure
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0.0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    print(f"True positives: {TP}\nTrue negatives: {TN}\nFalse positives: {FP}\nFalse negatives: {FN}")
    print(f"{80*'-'}\nprecision: {precision}\nrecall: {recall}\naccuracy: {accuracy}\nf_measure: {f_measure}\n{80*'-'}")
    # Return the calculated metrics


##########################################################################################################
if __name__ == "__main__":

    count_corpus_get_top_list = True
    count_corpus_get_top_list_only_content_words = False
    combine_two_lists = False
    compare_cognates_top_words = False
    random_draft_words = False
    evaluate_lat = False

    if not os.path.exists("../word_lists"):  # check if the folder exists, else create it
        os.makedirs("../word_lists")

    if count_corpus_get_top_list:
        time = "new"
        lang = "fr"
        top_n = 2000
        filename = f"../corpora/{lang}-{time}_tokenized.json"
        output_file = f"../word_lists/top{top_n}words_{lang}_{time}_with-pos2.txt"
        word_list = count_word_occurrences(filename, top_n, filter_pos=False, remove_stop=False)
        save_word_list_to_txt_file(word_list, output_file)

    if count_corpus_get_top_list_only_content_words:
        time = "new"
        lang = "es"
        top_n = 500
        filename = f"../corpora/{lang}-{time}_tokenized.json"
        output_file = f"../word_lists/top{top_n}words_{lang}_{time}_with-pos_only-content.txt"
        word_list = count_word_occurrences(filename, top_n, filter_pos=True)
        save_word_list_to_txt_file(word_list, output_file)

    if combine_two_lists:  # indicate the two files to be combined to a list
        lang = "fr"
        file1 = f'../word_lists/top2000words_{lang}_old_with-pos2.txt'
        file2 = f'../word_lists/top2000words_{lang}_new_with-pos2.txt'
        output_file = f'../word_lists/top2000_words_{lang}_with-pos2.txt'
        compare_two_files(file1, file2, output_file)

    if compare_cognates_top_words:
        cognate_file = "../word_lists/complete_cognate_list-en-lat-fr-es_manually-curated.txt"
        es_file = '../word_lists/top_words_es_with-pos_only-content.txt'
        fr_file = '../word_lists/top_words_fr_with-pos_only-content.txt'
        output_file = '../word_lists/cognate-list_en-fr-es_top-words.txt'
        compare_top_words_to_latin_list(cognate_file, es_file, fr_file, output_file)

    if random_draft_words:
        input_file = "../word_lists/top_words_en_with-pos_only-content.txt"
        output_file = "../word_lists/random_top_words_en_latin-detection-check.txt"
        number_of_words = 100
        random_list(input_file, output_file, number_of_words)

    if evaluate_lat:
        input_file = "../word_lists/latin_etymology_check/random_top_words_en_latin-detection-check_post.txt"
        tags = file_to_list(input_file, row=1)
        confusion_matrix(tags)


    # check_latin_detection()


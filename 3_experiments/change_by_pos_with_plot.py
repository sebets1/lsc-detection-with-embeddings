#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import os
import gensim
import matplotlib.pyplot as plt
from compare_en_with_without_latin_origin import get_cosines_of_list
import json


def count_word_occurrences(filename, top_n=500, filter_pos=False):
    """
    Take a tokenized and lemmatized corpus (as list of sentences) as input.
    Then count lemmas and return a ranked frequency list of the n most common words.
    """

    tot_count = 0
    function_count = 0
    pos_function = ["part", "aux", "cconj", "sconj", "adp", "pron", "det", "num"]

    word_collection = {}
    with open(filename, 'r', encoding='utf-8') as f:
        loaded_corpus = json.load(f)
        for sent in loaded_corpus:
            for token in sent:

                if token in word_collection:
                    word_collection[token] += 1
                else:
                    word_collection[token] = 1

    for entry in word_collection.items():
        if entry[0].split("_")[1] in pos_function:
            function_count += entry[1]
        tot_count += entry[1]

    ranked_words = sorted(word_collection.items(), reverse=True, key=lambda item: item[1])
    filtered_ranked_list = []
    for i, tuple_token in enumerate(ranked_words):
        if i <= top_n:
            filtered_ranked_list.append(tuple_token)
    return filtered_ranked_list, tot_count, function_count


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


def get_colors(lang):
    if lang == "en":
        col = ["cadetblue", "lightblue"]
        language = "English"
    if lang == "fr":
        col = ["darkgoldenrod", "wheat"]
        language = "French"
    if lang == "es":
        col = ["olivedrab", "#BCD193"]
        language = "Spanish"
    return col, language


def create_bar_chart(pos_list_mean_values, lang, output_name="test", choose_col="default"):
    """
    Input: The list with average cosine similarity scores for every part of speech
    Output: a nice bar chart
    """
    values = []
    pos_tags = []
    pos_content = ["adj", "adv", "noun", "verb", "propn", "intj"]
    pos_function = ["part", "aux", "cconj", "sconj", "adp", "pron", "det", "num"]
    for ele in pos_list_mean_values:
        values.append(ele[0])
        pos_tags.append(ele[1])

    # Create the bar chart
    plt.figure(figsize=(10, 6))  # Set figure size

    col, language = get_colors(lang)
    if choose_col != "default":
        col = choose_col

    colors = []
    for pos in pos_tags:
        if pos in pos_content:
            colors.append(col[0])  # Content words
        elif pos in pos_function:
            colors.append(col[1])  # Function words

    plt.bar(pos_tags, values, color=colors, edgecolor="dimgray")

    # Add labels and title
    plt.xlabel('PoS', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title(f'Cosine Similarity POS tags ({language})', fontsize=14)

    # Add legend
    content_patch = plt.Line2D([0], [0], color=col[0], lw=4, label='Content Words')
    function_patch = plt.Line2D([0], [0], color=col[1], lw=4, label='Function Words')
    plt.legend(handles=[content_patch, function_patch], loc='upper right', fontsize=10)

    # Add a grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value annotations on top of the bars
    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    # Set y-axis limits
    plt.ylim(0, 1.1)  # Ensure space for annotations above the bars

    plt.tight_layout()
    # Save the plot to a png image
    if not os.path.exists("../plots/part_of_speech"):  # check if the folder exists, else create it
        os.makedirs("../plots/part_of_speech")
    plt.savefig(f"../plots/part_of_speech/{lang}_pos-barplot_{output_name}.png", format="png", dpi=300)
    # close figure to release memory
    plt.close()


def create_boxplot(pos_list_scores, lang, output_name="test", choose_col="default"):
    """
    Input: Dictionary with cosine similarity scores for each part of speech
    Output: A boxplot showing distributions for content and function words
    """
    pos_content = ["adj", "adv", "noun", "verb", "propn", "intj"]
    pos_function = ["part", "aux", "cconj", "sconj", "adp", "pron", "det", "num"]

    # Prepare data for the boxplot
    means = {tag: sum(scores) / len(scores) for tag, scores in pos_list_scores.items()}
    sorted_pos_tags = sorted(pos_list_scores.keys(), key=lambda tag: means[tag])  # Sort by mean
    scores = [pos_list_scores[tag] for tag in sorted_pos_tags]

    col, language = get_colors(lang)
    if choose_col != "default":
        col = choose_col

    colors = [
        col[0] if tag in pos_content else col[1] if tag in pos_function else "gray"
        for tag in sorted_pos_tags
    ]

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(scores, patch_artist=True, labels=sorted_pos_tags, vert=True)

    # Style the boxplot
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('dimgray')

    # Add labels and title
    plt.xlabel('PoS', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title(f'Part of Speech Tags Cosine Similarity Distribution: {language}', fontsize=14)

    # Add legend
    content_patch = plt.Line2D([0], [0], color=col[0], lw=4, label='Content Words')
    function_patch = plt.Line2D([0], [0], color=col[1], lw=4, label='Function Words')
    plt.legend(handles=[content_patch, function_patch], loc='upper right', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the plot to a PNG image
    if not os.path.exists("../plots/part_of_speech"):  # Check if the folder exists, else create it
        os.makedirs("../plots/part_of_speech")
    plt.savefig(f"../plots/part_of_speech/{lang}_pos-boxplot_{output_name}1.png", format="png", dpi=300)

    # Close figure to release memory
    plt.close()


def compare_change_by_part_of_speech(wordlist, m1, m2, lang, plot):
    """
    Input: A list of top words to be analyzed and two aligned models.
    Output: A list and boxplot/bar plot showing the cosine similarity distributions for all words per part of speech.
    """
    words = file_to_list(wordlist)
    # words = wordlist
    output = wordlist.split("/")[-1][:-4]
    model1 = gensim.models.Word2Vec.load(m1)
    model2 = gensim.models.Word2Vec.load(m2)
    pos_tags = {}
    words_with_scores = get_cosines_of_list(words, model1, model2, exclude_propn=False)

    for ele in words_with_scores:
        pos = ele[1].split("_")[1]
        score = ele[0]
        if pos in pos_tags:
            pos_tags[pos].append(score)
        else:
            pos_tags[pos] = [score]

    if plot == "barplot":
        pos_mean_scores = []
        for pos in pos_tags.items():
            pos_count = len(pos[1])
            mean = sum(pos[1]) / pos_count
            pos_mean_scores.append([mean, pos[0], pos_count])
        create_bar_chart(pos_mean_scores, lang, output)

    if plot == "boxplot":
        create_boxplot(pos_tags, lang, output)
        print(f"Cosine similarity distributions for each part of speech tag in {lang}\n")
        for pos, scores in pos_tags.items():
            print(f"{pos}:\tMean = {round(sum(scores) / len(scores), 3)}, Count = {len(scores)}")
        print(60 * "-")

    if plot == "print":
        pos_mean_scores = []
        for pos in pos_tags.items():
            pos_count = len(pos[1])
            mean = sum(pos[1]) / pos_count
            pos_mean_scores.append([mean, pos[0], pos_count])

        pos_content = ["adj", "adv", "noun", "verb", "propn", "intj"]
        pos_function = ["part", "aux", "cconj", "sconj", "adp", "pron", "det", "num"]

        tot_content = []
        score_content = 0
        tot_function = []
        score_function = 0

        for ele in pos_mean_scores:
            if ele[1] in pos_content:
                tot_content.append(ele[0])
                score_content+=ele[0]
            if ele[1] in pos_function:
                tot_function.append(ele[0])
                score_function+=ele[0]

        mean_content = score_content/len(tot_content)
        mean_function = score_function/len(tot_function)
        diff = mean_function-mean_content
        print(50 * ".")
        print(f"Mean content words: {mean_content}")
        print(f"Mean function words: {mean_function}")
        print(f"Difference: {diff}")
        print(50*".")
        ###############################################################################
        pos_mean_scores.sort()
        #create_bar_chart(pos_mean_scores, lang, output)
        print(f"Average cosine similarities for each part of speech tag in {lang}\n")
        print("Word\tMean\tTotal Counts\n")
        for ele in pos_mean_scores:
            print(f"{ele[1]}:\t{round(ele[0], 3)}\t{ele[2]}")
        print(60 * "-")


######################################################################
if __name__ == '__main__':

    pos_plot = False
    pos_analysis = True
    pos_count = False

    plot_setting = "barplot"
    # plot_setting = "boxplot"
    # plot_setting = "print"

    if pos_plot:
        lang = "es"
        model1 = f"../models/aligned_model_{lang}-old_vec300_win10_mc5_ep5"
        model2 = f"../models/model_{lang}-new_vec300_win10_mc5_ep5"
        word_list = f"../word_lists/most_frequent_words/top1000_words_{lang}_with-pos2.txt"

        compare_change_by_part_of_speech(word_list, model1, model2, lang, plot_setting)

    if pos_counts:
        time = "old"
        lang = "en"
        top_n = 1000
        filename = f"../corpora/{lang}-{time}_tokenized.json"
        word_list1, tot1, function1 = count_word_occurrences(filename, top_n, filter_pos=False)
        time = "new"
        filename = f"../corpora/{lang}-{time}_tokenized.json"
        word_list2, tot2, function2 = count_word_occurrences(filename, top_n, filter_pos=False)

        matching_lines = []
        for ele in word_list1:
            for x in word_list2:
                if ele[0]==x[0]:
                    score = ele[1]+x[1]
                    matching_lines.append([ele[0], score])

        pos_tags = {}
        x_tags = []
        for ele in matching_lines:
            pos = ele[0].split("_")[1]
            if pos not in pos_tags:
                pos_tags[pos] = ele[1]
            else:
                pos_tags[pos] += ele[1]
        print(pos_tags)
        tot_count = tot1+tot2
        function_count = function2 + function1
        function_percent = 100 / tot_count * function_count
        print(tot_count)
        print(function_count)
        print(function_percent)



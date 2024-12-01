#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import random
import json
import re


def file_to_list(txt_file):
    """
    little script to return the first word of each line in a text file
    """
    wordlist = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("\n")[0]
            word = line.split("\t")[0]
            word = word.split(" ")[0]
            wordlist.append(word)
    return wordlist


def search_and_print_example_sent(sent_list, word, num_sent=10):
    found_sent = 0
    for ele in sent_list:
        if found_sent == num_sent:  # only print the chosen number of sentences, then stop searching
            break
        else:
            if word in ele[0]:
                if 5 < len(ele[0]) < 40:  # not have too short sentences but also not whole paragraphs
                    print(found_sent+1)
                    cleaned_sent = re.sub(r'\s+', ' ', ele[1])  # get rid of unnecessary whitespaces
                    print(cleaned_sent)
                    found_sent += 1
    print(80 * "-")


def get_examples(corpus1, corpus2, wordlist, num=10):
    """
    Goes through every word in list of words and prints n=num example sentences from each of the two corpora in which
    the word appears.
    """
    with open(corpus1, 'r', encoding='utf-8') as f:
        sent_list1 = json.load(f)
    random.shuffle(sent_list1)

    with open(corpus2, 'r', encoding='utf-8') as f:
        sent_list2 = json.load(f)
    random.shuffle(sent_list2)

    target_word = "caballero_noun"
    for word in wordlist:
        print(f"{num} example sentences for word {word}\n")
        print(f"from corpus {corpus1}\n")
        search_and_print_example_sent(sent_list1, word, num)
        print(f"from corpus {corpus2}\n")
        search_and_print_example_sent(sent_list2, word, num)


if __name__ == '__main__':

    lang = "en"
    corpus_old = f"../corpora/corpus_with_raw_sent/corpus_{lang}-old_tokenized.json"
    corpus_new = f"../corpora/corpus_with_raw_sent/corpus_{lang}-new_tokenized.json"
    word_list_file = f'../results/cognate_list_scores_sorted_{lang}.txt'

    word_list = file_to_list(word_list_file)
    word_list = word_list[0:3]

    print(word_list)
    num_of_examples = 10

    get_examples(corpus_old, corpus_new, word_list, num_of_examples)



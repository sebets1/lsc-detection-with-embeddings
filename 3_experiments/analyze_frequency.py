#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings


from find_most_changed import find_most_changed_words
from collections import Counter
import os
import json
from statistics import stdev
from statistics import variance


def build_corpus_book_separated(folder_path):
    """
    Put all json files in designated folder together as list of lists, each list being one json file /book from
    eltec corpus.
    """
    all_books = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):  # Process only JSON files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                book = []
                for sent in data:
                    for word in sent:
                        book.append(word)
                all_books.append(book)
    return all_books


def word_distribution(corpus, words):
    """
    Calculate word distribution across books.
    """
    word_count = {word: 0 for word in words}
    for book in corpus:
        unique_words_in_book = set(book)
        for word in words:
            if word in unique_words_in_book:
                word_count[word] += 1
    return word_count


def total_frequency(corpus, words):
    """
    Calculate total frequency across all books.
    """
    word_freq = Counter()
    for book in corpus:
        word_freq.update(book)
    return {word: word_freq[word] for word in words}


def word_frequencies_per_book(corpus, words):
    """
    Function to calculate word frequencies in each book.
    """
    word_frequencies = {word: [] for word in words}
    for book in corpus:
        book_counter = Counter(book)  # Count occurrences of words in the book
        book_tot = len(book)
        for word in words:
            tf = book_counter[word]/book_tot*10000
            word_frequencies[word].append(round(tf, 3))  # Append the count for the word
    return word_frequencies


if __name__ == '__main__':
    lang = "es"
    model1 = f"../models/model_{lang}-new_vec300_win10_mc5_ep5"
    model2 = f"../models/aligned_model_{lang}-old_vec300_win10_mc5_ep5"
    word_list = f'../word_lists/top_words_{lang}_with-pos_only-content.txt'

    folder_path = f"../corpora/{lang}-novels/"

    most_changed = find_most_changed_words(word_list, model1, model2, top_n=20)

    most_changed_words = [ele[0] for ele in most_changed]

    corpus1 = build_corpus_book_separated(f"{folder_path}old")
    corpus2 = build_corpus_book_separated(f"{folder_path}new")

    # Compute distribution for both corpora
    dist1 = word_distribution(corpus1, most_changed_words)
    dist2 = word_distribution(corpus2, most_changed_words)

    # Combine distributions
    combined_distribution = {word: (dist1[word], dist2[word]) for word in most_changed_words}

    # Print results
    print("Word distribution across books:")
    for word, (count1, count2) in combined_distribution.items():
        print(f"{word}: Corpus1 - {count1} books, Corpus2 - {count2} books")

    # Compute total frequency for both corpora
    freq1 = total_frequency(corpus1, most_changed_words)
    freq2 = total_frequency(corpus2, most_changed_words)

    # Combine frequency and distribution data
    for word in most_changed_words:
        print(f"{word}:")
        print(f"  Corpus1 - {dist1[word]} books, {freq1[word]} total occurrences")
        print(f"  Corpus2 - {dist2[word]} books, {freq2[word]} total occurrences")


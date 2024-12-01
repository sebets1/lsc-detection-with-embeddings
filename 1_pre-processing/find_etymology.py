#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings


import requests
from bs4 import BeautifulSoup
import Levenshtein


def get_etymology(word):
    """
    Scrape the etymonline.com for the etymology of a word.
    """
    url = f"https://www.etymonline.com/word/{word}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the etymology text
    etymology_section = soup.find("section", {"class": "word__defination--2q7ZH"})
    if etymology_section:
        return etymology_section.get_text()
    else:
        return None


def extract_latin_origin(etymology):
    """
    Check if word seems to come from latin origin and if true, returns the entry with the latin word it is derived
    from. If no latin origin found in etymology, then returns None, so word will not be added to latin_origin_words dict.
    """
    if "from Latin" in etymology:
        start = etymology.find("from Latin")
        # Find the next period or comma to get the Latin word
        end = etymology.find(",", start)
        if end == -1:
            end = etymology.find(".", start)
        # Extract and clean the Latin phrase or word
        latin_phrase = etymology[start:end].strip()
        return latin_phrase
    return None


def check_if_really_latin(word, latin_origin):
    """
    As the scraping method is not always reliable, additionally find levenshtein distance. If it is bigger than XX,
    print the word in terminal with a warning to manually check it.
    """
    x = latin_origin.split(" ")
    if len(x)>=3:
        latin_word = x[2]
        distance = Levenshtein.distance(word, latin_word)
        if distance >= len(word):
            print(f"\nManual check needed:\nIs '{word}' really derived from Latin '{latin_word}'?")
            print(f"Etymology: {latin_origin}\n")
    else: # some mistake in pre-processing. I think 'body' is always doing weird.
        print(x)
        print(word)


def find_words_with_latin_roots(word_list, with_pos=False):
    """
    Gets the etymology from etymonline for each word in the input word list via function get_etymology().
    Checks if word of latin origin and if true searches the original latin word via function extract_latin_origin().
    Returns a dictionary with all English words of latin origin as key and the latin word they're derived from as value.
     """
    latin_origin_words = {}

    for word in word_list:
        if with_pos:
            token = word.split("_")[0]
        else:
            token = word
        etymology = get_etymology(token)
        if etymology:
            latin_origin = extract_latin_origin(etymology)
            if latin_origin:
                latin_origin_words[word] = latin_origin
                check_if_really_latin(token, latin_origin)
    print(f"Total words found with latin roots in list of most frequent words: {len(latin_origin_words)}")
    return latin_origin_words


def save_dict_to_txt_file(word_dictionary, output_file):
    """
    Saves the words from the latin origin list in a text file for further use. Each line in the text file is in the
    format <English word> <tab> <Latin word>.
    """
    with open(output_file, "w", encoding='utf-8') as file:
        for key, value in word_dictionary.items():
            latin_word = value.split(" ")
            if len(latin_word) > 2: # neccessarry add-on as there seems to sometimes be one word which isn't processed properly
                latin_word = latin_word[2]
                file.write(f"{key}\t{latin_word}\n")
            else:
                print(latin_word)


if __name__ == '__main__':

    # if the words in top word list have a pos tag (e.g. "say_verb"), remove them for etymology search
    token_with_pos = True

    # adapt input and output file
    input_file = "../word_lists/top_words_en_with-pos_only-content.txt"
    output_file = "../word_lists/en_top-words_with_latin_roots_only-content.txt"

    ###########################################################################################
    wordlist = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            word = line.split("\n")[0]
            wordlist.append(word)

    latin_words = find_words_with_latin_roots(wordlist, with_pos=token_with_pos)
    save_dict_to_txt_file(latin_words, output_file)






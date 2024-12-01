#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import json
import random
import spacy
import os


def preprocess(filename, language, add_raw_sent=False):
    """
    Read in the data from the text file and tokenize for Word2Vec as list containing sentence lists of lemmas
    with PoS tags.
    If add_raw_sent is chosen, the first part of each list is the processed sentence as list of lemmas, the
    second part is the raw sentence. Not adapted for other scripts (like train_models.py) but needed for the
    script 'get_example_sentences.py'.
    """
    # Load the raw text file for tokenization
    with open(filename, 'r', encoding='utf-8') as f:
        raw_text = ""
        for line in f:
            raw_text += line.replace("\n", " ")

    if language=="en":
        nlp = spacy.load('en_core_web_sm')
    if language=="fr":
        nlp = spacy.load('fr_core_news_sm')
    if language=="es":
        nlp = spacy.load('es_core_news_sm')

    # default max char length in spacy is 1 mio to avoid memory problems with NER or dependency parsing
    nlp.max_length = 3000000

    # split the raw text into manageable chunks to not overload memory (adapt setting to own machine)
    # (most novels are not that big, but a few are)
    chunk_size = 1000000
    chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]

    tokenized_corpus = []

    for i, chunk in enumerate(chunks):
        print(f"Started chunk {i+1}.")
        doc = nlp(chunk)
        # tokenize text
        for sent in doc.sents:
            # Create a lemmatized list of tokens with POS tags, excluding punctuation and spaces
            lemma_pos_sent = [
                f"{token.lemma_.lower()}_{token.pos_}".lower()
                for token in sent if not token.is_punct and not token.is_space
            ]
            if len(lemma_pos_sent) != 0:
                if add_raw_sent == True:
                    tokenized_corpus.append([lemma_pos_sent, sent.text])
                else:
                    tokenized_corpus.append(lemma_pos_sent)
        print(f"Chunk {i + 1} finished. {len(chunks) - i - 1} left.")

    if not tokenized_corpus:
        raise Exception("Could not preprocess. Maybe check language settings.")

    print(f"finished tokenizing\n{40*'-'}\n")
    return tokenized_corpus


def process_all_eltec_files(with_raw_sent=False):
    """
    Goes through all ELTeC novels, processes them and saves them as lemmatized lists in JSON files.
    Furthermore, puts together all novels of the same epoch and language into one big corpus and shuffles sents.
    """
    lang_list = ["es", "fr", "en"]
    lang_list = [ "fr", "en"]
    time = ["old", "new"]

    for lang in lang_list:
        for t in time:
            folder_path = f"../corpora/{lang}-novels/{t}"
            corpus_tot = []
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):  # Process only txt files
                    file_path = os.path.join(folder_path, filename)
                    output_file = f"{folder_path}/processed_{filename[4:-4]}.json"

                    print(f"Pre-process {filename}...")
                    data = preprocess(file_path, lang, add_raw_sent=with_raw_sent)
                    with open(output_file, 'w', encoding='utf-8') as file:
                        json.dump(data, file, ensure_ascii=False, indent=4)
                    for sent in data:
                        corpus_tot.append(sent)
            random.shuffle(corpus_tot)  # shuffle all sentences - erase bias towards later documents in data
            output_corpus = f"../corpora/corpus_{lang}-{t}_tokenized.json"
            with open(output_corpus, 'w', encoding='utf-8') as file:
                json.dump(corpus_tot, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    #################################################################################
    # Tokenize and lemmatize a corpus stored in a raw text file and save as JSON file.
    #################################################################################

    # choose settings
    single_file = False
    process_all_eltec = True  # needed for model training and some experiments
    process_all_eltec_with_raw_sent = False  # needed for "get_example_sentences.py

    ################################################################################
    if process_all_eltec:
        process_all_eltec_files()

    if process_all_eltec_with_raw_sent:
        process_all_eltec_files(with_raw_sent=True)

    ################################################################################
    # adapt file paths and language
    if single_file:
        lang = "en"
        input_file = f"../corpora/raw-text_{lang}-novels_new_corpus.txt"
        output_file = f"../corpora/{lang}-new_tokenized.json"
        data = preprocess(input_file, lang)
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)












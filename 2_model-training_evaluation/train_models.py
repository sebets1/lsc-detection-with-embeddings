#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import os
from gensim.models import Word2Vec
from collections import Counter
import json


def train_word2vec_model(preprocessed_corpus, vector_dim=200, context_window=5, min_occurrences=5, epoch_num=5,
                         save_model="no"):
    """
    Trains a word2vec model with my chosen parameters on the provided corpus and saves it for further use.
    """
    print("start training")
    # vector size: 300 is used in most papers, window size: 5 is very common, sg=1 for skip-gram
    # min_count: minimum number of appearances a word should have to be included in vocab
    # epochs: default epochs 5 is used in most papers on the topic
    # workers: speeds up training, depends on no. of cpu cores available, doesn't matter much for me as corpora small
    # random seed: use the same random seed (negative sampling) to have more consistent, more comparable embeddings
    w2v_model = Word2Vec(sentences=preprocessed_corpus, vector_size=vector_dim, window=context_window,
                         min_count=min_occurrences, epochs=epoch_num, workers=4, sg=1, seed=42)

    ################################################################################################################
    # for information: gensim's word2vec default settings:
    """
    classgensim.models.word2vec.Word2Vec(sentences=None, corpus_file=None, vector_size=100, alpha=0.025, window=5,
                                         min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3,
                                         min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1,
                                         epochs=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, 
                                         compute_loss=False, callbacks=(), comment=None, max_final_vocab=None)
    """
    #################################################################################################################
    print("end training")

    if save_model != "no": # save model if chosen
        # save_model_name = f"../models/model_{save_model}_vec{vector_dim}_win{context_window}_mc{min_occurrences}"
        save_model_name = f"../models/model_{save_model}_vec{vector_dim}_win{context_window}_mc{min_occurrences}_ep{epoch_num}"
        if not os.path.exists("../models"):  # check if the folder exists, else create it
            os.makedirs("../models")
        w2v_model.save(save_model_name)

    return w2v_model


def analyse_vocab_size(tokenized_corpus):
    """
    Analyse my corpus and check how big my vocabulary would be with different minimum counts per word in vocab.
    """
    # Calculate word frequencies
    word_freq = Counter([word for sentence in tokenized_corpus for word in sentence])

    # Check how many words would be excluded based on different min_count thresholds
    for min_count in [1, 2, 3, 4, 5, 10, 20]:
        vocab_size = sum(1 for word, freq in word_freq.items() if freq >= min_count)
        print(f"Words with frequency >= {min_count}: {vocab_size}")

######################################################################################################################
if __name__ == '__main__':

    # Analyse the size of the vocabulary of a corpus with different minimum occurrence counts per lemma
    analyse_vocabulary = False
    # Train embedding model
    train = False
    train_all_eltec = True

    # Choose data
    name = "en-old"
    json_file = f"../corpora/corpus_{name}_tokenized.json"
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Adapt hyperparameters for model training
    vector_dimension = 300
    window = 10
    min_count = 5
    epochs = 5

######################################################################################################################

    if analyse_vocabulary:
        analyse_vocab_size(data)

    if train:
        embedding_model = train_word2vec_model(preprocessed_corpus=data, vector_dim=vector_dimension,
                                               context_window=window, min_occurrences=min_count, epoch_num=epochs,
                                               save_model=name)

    if train_all_eltec:
        lang = ["es", "fr", "en"]
        time = ["old", "new"]
        for l in lang:
            for t in time:
                name = f"{l}-{t}"
                json_file = f"../corpora/corpus_{name}_tokenized.json"
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                embedding_model = train_word2vec_model(preprocessed_corpus=data, vector_dim=vector_dimension,
                                                       context_window=window, min_occurrences=min_count,
                                                       epoch_num=epochs,
                                                       save_model=name)


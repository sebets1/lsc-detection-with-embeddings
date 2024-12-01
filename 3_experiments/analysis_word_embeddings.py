#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import sys
import os
import gensim
import networkx as nx
import matplotlib.pyplot as plt


def plot_word_neighbors(model, target_word, model_name="test", color="lightblue"):
    """
    Creates a plot with the target word in the middle and around it words whose vectors have the highest cosine
    similarity to the target word.
    """
    neighbors = model.wv.most_similar(target_word, topn=20)

    ###################################################
    # to just print closest neighbors with the same pos
    # for ele in neighbors:
    #     context_word = ele[0].split("_")[0]
    #     pos = ele[0].split("_")[1]
    #     if pos == target_word.split("_")[1]:
    #         print(context_word)
    ###################################################
    G = nx.Graph()
    G.add_node(target_word)

    # add neighbors as nodes and connect them to the target word
    for neighbor, similarity in neighbors:
        G.add_node(neighbor)
        G.add_edge(target_word, neighbor, weight=similarity)

    # Draw the graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)  # Position nodes with a force-directed algorithm

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color=color, font_size=12, node_size=2000, edge_color='gray')

    # Save the plot to a png image
    if not os.path.exists("../plots/closest_neighbor"):  # check if the folder exists, else create it
        os.makedirs("../plots/closest_neighbor")
    plt.savefig(f"../plots/closest_neighbor/{target_word}_{model_name}.png", format="png", dpi=300)
    # close figure to release memory
    plt.close()



if __name__ == '__main__':

    all_plots_in_cognates_list = True
    all_plots_in_word_list = False
    single_word_plot = False

    wordlist = "../word_lists/cognate-list_en-fr-es_top-words.txt""
    single_word = "computer_noun"

    lang = "fr"
    model1 = f"../models/aligned_model_{lang}-old_vec300_win10_mc5_ep5"
    model2 = f"../models/model_{lang}-new_vec300_win10_mc5_ep5"

##############################################################################################3

    m1 = gensim.models.Word2Vec.load(model1)  # import already created models
    m2 = gensim.models.Word2Vec.load(model2)

    model_name1 = model1.split("/")[-1]  # to save plotted figure with the model name
    model_name2 = model2.split("/")[-1]

    # give plots from the older and newer corpora different colors to distinguish them
    plot_color1 = "lightblue"
    plot_color2 = "orchid"
    plot_color3 = "lightgreen"

    #######################
    if all_plots_in_cognates_list:

        with open(wordlist, 'r', encoding='utf-8') as file:
            for line in file:
                if lang == "en":
                    word = line.split("\t")[0]
                    #word = line.split(":")[0]
                if lang == "fr":
                    word = line.split("\t")[2]
                if lang == "es":
                    word = line.split("\t")[3]
                    word = word.replace("\n", "")
                plot_word_neighbors(m1, word, model_name1, plot_color1)
                plot_word_neighbors(m2, word, model_name2, plot_color3)
                print(f"Closest neighbors plot done for word '{word}'.")

    #######################
    if all_plots_in_word_list:
        with open(wordlist, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.split("\t")[0]
                word = word.replace("\n", "")
                plot_word_neighbors(m1, word, model_name1, plot_color1)
                plot_word_neighbors(m2, word, model_name2, plot_color3)
                print(f"Closest neighbors plot done for word '{word}'.")

    #######################
    if single_word_plot:
        plot_word_neighbors(m1, single_word, model_name1, plot_color1)
        plot_word_neighbors(m2, single_word, model_name2, plot_color2)
        print(f"Closest neighbors plot done for word '{single_word}'.")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import gensim
import numpy as np


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Code from: https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8

    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.

    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words) # if I pass a word list for words parameter, it will only keep vocabulary appearing in the word list

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)

    return other_embed


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key

        print(len(m.wv.key_to_index), len(m.wv.vectors))

    return (m1, m2)


####################################################################
# my code starts here:
####################################################################
def align_and_save(base_model, other_model, output):
    """
    Input: Two models whose vector spaces should be aligned.
    Output: New model 2, which is aligned to model 1.
    """
    embedding_model_base = gensim.models.Word2Vec.load(base_model)
    embedding_model_other = gensim.models.Word2Vec.load(other_model)
    aligned_model_other = smart_procrustes_align_gensim(embedding_model_base, embedding_model_other)
    aligned_model_other.save(output)

    # test alignment
    # vector_table1 = embedding_model_base.wv['terrible']
    # vector_table2 = aligned_model_other.wv['terrible']
    # cosine_similarity = np.dot(vector_table1, vector_table2) / (np.linalg.norm(vector_table1) *
    #                                                             np.linalg.norm(vector_table2))
    # print(f"Cosine Similarity between 'terrible' embeddings: {cosine_similarity}")


if __name__ == '__main__':

    # choose settings
    align_two_models = False
    align_all_eltec = True

    if align_two_models:
        # set the two Word2Vec models to align
        base_embed = "../models/model_en-new_vec300_win10_mc5_ep5"
        other_embed = "../models/model_en-old_vec300_win10_mc5_ep5"

        output_model = f"../models/aligned_{other_embed.split('/')[-1]}"
        align_and_save(base_embed, other_embed, output_model)

    if align_all_eltec: # check if model names are correct
        lang = ["fr", "es", "en"]
        for l in lang:
            base_embed = f"../models/model_{l}-new_vec300_win10_mc5_ep5"
            other_embed = f"../models/model_{l}-old_vec300_win10_mc5_ep5"

            output_model = f"../models/aligned_{other_embed.split('/')[-1]}"
            align_and_save(base_embed, other_embed, output_model)






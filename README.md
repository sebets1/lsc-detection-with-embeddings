# Detecting Semantic Change with Word Embeddings

The following project was done as part of my bachelor thesis titled "Multilingual Semantic Change Detection with 
Word Embeddings in 19th Century Novels" in 2024 at the Computational Linguistics 
Institute from the University of Zurich. Find the abstract at the bottom of this README.

### Used data
The project was conducted using the French, Spanish and English collections of the open-source ELTeC corpus, available
under: https://github.com/COST-ELTeC/ELTeC

### Requirements
The required python libraries and extensions are listed in the requirements.txt file.


## How to work with this repository

Following is an overview of all scripts in this repository. Note that most scripts serve multiple functions. 
The desired function is chosen directly in the script at the bottom. Check if the required files to run the script 
were already created and are in the right folders.
Some of the generated word lists, results and plots are provided too. 
Due to file size restrictions, the models and some of the data could not be included but are all 
reproducible with the provided scripts.


### 1 Pre-processing
Contains 4 scripts used to prepare the data for model training.
- `save_xml_books_as_txt.py`: Extracts all novels sorted by date from the ELTeC xml files
- `pre_process_corpora.py`: Pre-processing of each file (lemmatization, POS-tagging etc.), preparation for model training
- `prepare_target_words.py`: Creates word lists necessary for the different experiments
- `find_etymology.py`: Automatically extracts those English words from a list which contain Latin roots

### 2 Model training and evaluation
Contains 3 scripts for training, alignment and measuring of word embeddings.
- `train_models.py`: Trains word2vec embedding models 
- `procrustes_align.py`: Aligns two models using orthogonal Procrustes to make the vector spaces comparable
- `cosine_similarity.py`: Calculates the cosine similarity between two aligned embeddings for each word in a provided list


### 3 Experiments
Contains 8 scripts for different experiments and analyses.
- `analysis_word_embeddings.py`: Returns nearest neighbours of a target word and creates plots
- `analyze_frequency.py`: Analyzes in how often and in how many different novels target words appear
- `compare_change_by_pos.py`: Finds the mean cosine similarities of parts-of-speech and prints them
- `change_by_pos_with_plot.py`: Finds the mean cosine similarities of parts-of-speech and plots them
- `cognates_spearman_evaluation.py`: Calculates Spearman correlation between cognates of two languages
- `compare_en_with_without_latin_origin.py`: Analyzes the measured change of English words with and without Latin origin
- `find_most_changed.py`: Finds the words with largest/smallest cosine similarity from a word list
- `get_example_sentences.py`: Prints random example sentences from both corpora of a language which include the target word(s)


-------------------------------------

## Abstract

Recent years have seen a rise in using computational methods to detect semantic
change in words. This thesis explores the use of static word embeddings to measure
semantic change and compare it cross-linguistically. I propose a method for identifying
English words of Latin origin and manually curate a small dataset of cognates
in French, Spanish and English. I measure semantic change using cosine similarity
and find that English words of Latin origin undergo a 3% larger semantic change
compared to words without Latin roots. Further, I examine the curated cognate sets
and report a strong correlation between the measured semantic change of French
and Spanish words. Finally, I show that parts-of-speech exhibit similar patterns of
change across the three examined languages, with function words being more stable
than content words, and proper nouns exhibiting the largest measured change.



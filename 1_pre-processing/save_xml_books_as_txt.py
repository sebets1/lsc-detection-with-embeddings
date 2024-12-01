#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Seraina Betschart
# date: 01.12.2024
# Bachelor Thesis
# Detecting Semantic Shift with Word Embeddings

import os
import xml.etree.ElementTree as ET


def build_corpora(folder_path, output_file="test", cut1=1875, cut2=1885):
    """
    Goes through all XML files in the specified folder. Filters out raw text (only the novel, without context info)
    and appends it to one of two text files regarding the specified time period cut-off.
    """

    if not os.path.exists(f"../corpora/{output_file}/old"):  # check if the folder exists, else create it
        os.makedirs(f"../corpora/{output_file}/old")
    if not os.path.exists(f"../corpora/{output_file}/new"):  # check if the folder exists, else create it
        os.makedirs(f"../corpora/{output_file}/new")

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):  # Process only XML files
            file_path = os.path.join(folder_path, filename)
            date = get_date(file_path)
            raw_text = extract_text_from_xml(file_path)

            if (int(date)) <= cut1:
                save_file = f"../corpora/{output_file}/old/raw_{filename[:-4]}.txt"
                with open(save_file, "w", encoding='utf-8') as file:
                    file.write(raw_text)
            if (int(date)) >= cut2:
                save_file = f"../corpora/{output_file}/new/raw_{filename[:-4]}.txt"
                with open(save_file, "w", encoding='utf-8') as file:
                    file.write(raw_text)

    print(f"Finished corpora {output_file}")


def get_date(filename):
    """
    Finds the year in which the novel was published. Different method for ENG, as these files have the year
    conveniently stored in the filename.
    """
    string_list = filename.split("/")
    if string_list[-1][0:3] == "ENG":
        date=string_list[-1][3:7]
        return date
    if string_list[-1][0:3] == "FRA":
        date=extract_date_from_xml(filename)
        return date
    if string_list[-1][0:3] == "SPA":
        date=extract_date_from_xml(filename)
        return date


def extract_date_from_xml(filename):
    """
    Function for French and Spanish novels where date is not in the file name. Finds date of novel by accessing
    the <date> element within the <bibl> element with type="firstEdition".
    """
    with open(filename, 'r', encoding='utf-8') as file:
        xml_content = file.read()
        # Parse the XML content

    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}  # Define the namespace

    # Parse the XML content
    root = ET.fromstring(xml_content)

    # Locate the <bibl> element with type="firstEdition"
    bibl_elements = root.findall('.//tei:bibl[@type="firstEdition"]', namespace)

    if bibl_elements:
        # Iterate through found elements in case there are multiple
        for bibl_element in bibl_elements:
            # Locate the <date> element within the <bibl> element
            date_element = bibl_element.find('tei:date', namespace)

            # Get the text content of the <date> element
            if date_element is not None:
                first_edition_date = date_element.text
                if len(first_edition_date) > 4:  # A few novels have a time span of two years.
                    first_edition_date = first_edition_date[0:4]
                return first_edition_date

    else:
        print(f"not found for {filename}")
        return f"not found for {filename}"


def extract_text_from_xml(filename):
    """
    Extract data from XML string
    """
    with open(filename, 'r', encoding='utf-8') as file:
        xml_content = file.read()
    # Parse the XML content
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    root = ET.fromstring(xml_content)

    # Extract raw text from the <text> element
    body_element = root.find('.//tei:text', namespace)
    raw_text = ""
    if body_element is not None:
        raw_text = ET.tostring(body_element, encoding='unicode', method='text')

    return raw_text


def pre_analysis(folder_path, cut1=1875, cut2=1885):
    """
    Goes through all files in folder. Returns total token size and number of documents/files for two defined time
    periods. This helps to find an appropriate time cut-off.
    """
    lang = folder_path[-9:-7]
    word_count_c1 = 0
    no_of_novels_c1 = 0
    word_count_c2 = 0
    no_of_novels_c2 = 0
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):  # Process only XML files
            file_path = os.path.join(folder_path, filename)
            date = get_date(file_path)
            raw_text = extract_text_from_xml(file_path)

            if len(date) > 4:
                pass
            elif (int(date)) <= cut1:
                no_of_novels_c1 += 1
                word_count_c1 += len(raw_text.split())
            elif (int(date)) >= cut2:
                no_of_novels_c2 += 1
                word_count_c2 += len(raw_text.split())

    print(f"Analysis for {lang}")
    print(40*"-")
    print(f"Total words for old novels: {word_count_c1}")
    print("No. of old novels: "+ str(no_of_novels_c1))
    print(f"Total words for new novels: {word_count_c2}")
    print("No. of new novels: " + str(no_of_novels_c2))
    print(40*"-"+"\n\n")


def single_test_file(file_path="../data/en-novels/ENG18400_Trollope.xml"):
    """
    Process a single XML file and check raw text, published date and total number of words.
    """
    date = get_date(file_path)
    # Extracting data
    raw_text = extract_text_from_xml(file_path)
    # print(raw_text)
    print(f"\n{40*'-'}\ndate: {date}")
    print(f"number of words: {len(raw_text.split())}")


if __name__ == '__main__':
    # Choose setting
    single_file = False
    analyse_xml = False
    extract_raw_text = True

    # Decide at which years to split corpora. (ELTeC corpora distributed across 1840-1920)
    cut_off1 = 1875
    cut_off2 = 1885

    ################################################################################################
    folder_path_list = ["../data/en-novels", "../data/fr-novels", "../data/es-novels"]

    if single_file:
        single_test_file(file_path="../data/en-novels/ENG18400_Trollope.xml")

    if analyse_xml:
        for ele in folder_path_list:
            pre_analysis(ele, cut1=cut_off1, cut2=cut_off2)


    if extract_raw_text:
        for ele in folder_path_list:
            output_file = ele[-9:]
            build_corpora(ele, output_file)







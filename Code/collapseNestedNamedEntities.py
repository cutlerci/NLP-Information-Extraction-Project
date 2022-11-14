#########################################################
#   VCU CMSC 516 Advanced Natural Language Processing   #
#           Information Extraction Project              #
#                                                       #
#           Developed by Charles Cutler,                #
#      Christopher Samson, and Majd Alkawaas            #
#                                                       #
#                October 30, 2022                       #
#########################################################

# The following code was developed for the second programming assignment in the course
# CMSC 516, Advanced Natural Language Processing, at Virginia Commonwealth University
#
# If you do not use Google Colab, make sure to install these python libraries.
# Installation instructions can be found at:
#
# https://pandas.pydata.org/docs/getting_started/install.html

import os
import pandas as pd

# Recursive Function to collapse the nested named entitiy layers into one later
# it returns the entity at the highest or (least nested) layer
def highestTaggedNestLayer(layers, highestLayerIndex):
    """
    Collapses the nested named entity layers found in the LitBank Dataset.
    :param layers, highestLayerIndex: A list of the named entity layers for a given token, The index of the highest layer in the list
    :return entity: Returns the named entity found at the highest layer that is not an "O"
    """
    if highestLayerIndex == -1:
        return layers[0]

    if highestLayerIndex >= 0 and layers[highestLayerIndex] == 'O':
        return highestTaggedNestLayer(layers, (highestLayerIndex-1))
    else:
        return layers[highestLayerIndex]

# Sets the directory of the original dataset files
directory = 'dbamman litbank master entities-tsv'

# Gets all of the file names that are not hidden files ( such as .DSTORE )
filesToCollapse = [item for item in os.listdir(directory) if not item.startswith('.')]

# For every file in the directory we are going to collapose the named entity tags. 
# This is done for each word in every file 
collapsedRows = []
numberOfSentences = 1

for filename in filesToCollapse:
    print(f'Started: {filename} ')

    # Get the input file pathfor the next file in the directory
    inputFileName = os.path.join(directory, filename)
    # Read the lines of the input file
    with open(inputFileName) as book:
        out = book.readlines()
    
    numberOfColumns = len((out[0].rstrip('\t\n')).split('\t')) + 1

    makeDataFrame = range(numberOfColumns)
    df = pd.DataFrame(columns=makeDataFrame)

    
    index = 0
    for item in out:
        columns = []

        if item == '\n':
            numberOfSentences += 1
            continue
        else:
            columns.append(str(numberOfSentences))
              
        for element in (item.rstrip('\t\n')).split('\t'):
            columns.append(element)

        try:
            df.loc[index] = columns
        except:
            print(columns)


        index += 1

    # Get the column names and remove the last column due to an extra tab found on the end of every line in every file
    columnNames = list(df.columns)

    # For every row in a file collapse the named entities and store the collapsed version as a list of lists
    for index in df.index:
        layers = []
        for columnName in columnNames[2:]:
            layers.append(df[columnName][index])

        collapsedRows.append([df[0][index], df[1][index], highestTaggedNestLayer(layers, (len(layers)-1))])

    print(f'Finished: {filename}\n')
    
# Output the collasped Data to a .csv file using pandas
header = ["Sentence","Word", "Named Entity"]
cleanedDataFrame = pd.DataFrame(collapsedRows, columns=header)
cleanedDataFrame.to_csv("ProcessedLitBankDataset.csv", index=False)


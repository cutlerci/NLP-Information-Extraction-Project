![VCU Logo](https://ocpe.vcu.edu/media/ocpe/images/logos/bm_CollEng_CompSci_RF2_hz_4c.png)

# NLP-Information-Extraction-Project
| Developer Name | VCU Email Address | Github Username |
| :---: | :---: | :---: |
| Charles Cutler | cutlerci@vcu.edu | cutlerci |
| Christopher Smith | samsoncr@vcu.edu | samsoncr |
| Majd Alkawaas | alkawaasm@vcu.edu | MajdAlkawaas |

# For creating the table of contents
http://ecotrust-canada.github.io/markdown-toc/

# Project description

The objective of this project is to perform information extraction by applying a trained model to tweets we pull from Twitter. We were allowed to train our model to complete the information extraction task with a dataset of our choice. Information extraction is a classification problem in which our code will try to determine whether a word or symbol is considered a named entity. Named entities in our dataset cover six of the ACE 2005 classes in text:
* People (PER): *Tom Sawyer*, *her daughter*
* Facilities (FAC): *the house*, *the kitchen*
* Geo-political entities (GPE): *London*, *the village*
* Locations (LOC): *the forest*, *the river*
* Vehicles (VEH): *the ship*, *the car*
* Organizations (ORG): *the army*, *the Church*

A named entity can be of the 6 above classes and is preceded either by a "B-" or an "I-". A "B-" denotes that the word is the beginning of a named entity and an "I-" denotes that the word is an inclusive member of the named entity. If a word or symbol is not considered any of the above entities it is marked with an *O*. Given any sequence of words or symbols, referred to as tokens, our model will label each token as one of the named entity classes or as an *O*. Then we must analyze the accuracy of our model. We do this by comparing the model’s predictions to labels made by humans which are assumed to be the correct classification. The model for information extraction is a collection of Bidirectional Long Short Term Memory Units ( BiLSTM ) with a Conditional Random Field ( CRF ) layer for classification. 

The whole process is a series of steps beginning by preprocessing a large training dataset. The data used for this project can be found in the Data sub-directory. The dataset contains sentences from 100 literary works in which every token has been classified. These sentences must be preprocessed before being vectorized and embedded. After its transformation, the data can then be given to the BiLSTM + CRF model. The model will learn from the data how to predict whether a token is a named entity or not and, if so, what named entity it is. We then collect the predictions of our model on a test set of data, and for our results we calculate the precision, recall, F1, and accuracy scores of our predictions.

Each of these steps are explained in further detail in the sections below.

# Installation instructions

Begin by downloading the __________ file. You can follow this tutorial for downloading a single file from github: 

<small><i><a href='https://www.wikihow.com/Download-a-File-from-GitHub'>Downloading a Single File From GitHib</a></i></small>

Follow the same instructions for downloading the cleaned data set file called ProcessedLitBankDataset.csv found in the Data Sub Folder.

We will also need the glove.6B.200d.txt file which can be downloaded from 
<small><i><a href='https://www.kaggle.com/datasets/incorpes/glove6b200d'>glove.6b.200d.txt</a></i></small>
 
This file may be too big to download, however, so you can use this public Google Drive link https://drive.google.com/file/d/1VGGnJMZZgBG0_MCqb2mafvzgMT1NDJ1i/view?usp=share_link to upload the file to your drive. Click the link and use ZIP Extractor to extract the file to Google Drive.


<p align="center">
 <img src="./ZIP Extractor.png" width="750" height="500">
</p>

You can skip the downloads of  __________ and ProcessedLitBankDataset.csv if you already have downloaded the whole project as a zip file. This can be done by clicking on the green button that says "code" in the main directory of the github page (same directory as this README), and then clicking on "Download ZIP".

We recommend using Google Colab to run the code. To do so, go to Google Drive at drive.google.com. From your downloads on your local machine, upload the two files, __________ and ProcessedLitBankDataset.csv, to your Google Drive. 

Do the same for glove.6B.200d.txt either from your own downloads or the previous link. Make sure they are uploaded to your MyDrive folder in Google Drive.

If you choose to run the code locally, you must install Python and the necessary packages for TensorFlow, Numpy, Keras, and SKLearn.
Tutorials for installing Python and these libraries can be found at these links:

https://realpython.com/installing-python/

https://www.tensorflow.org/install 

https://numpy.org/install/ 

https://www.tutorialspoint.com/keras/keras_installation.htm 

https://scikit-learn.org/stable/install.html 

# Usage instructions

Open the __________ file in your Google Drive. There are two lines of code that need to be uncommented before running in Google Colab. These are lines __________ and __________. You can also read through the comments for more guidance through this process.

Then click on the runtime tab, and click run all.

If you chose to run the file locally, you will need to instead replace the file path names on lines __________ and __________ with the local file paths. This also means that you will have to download the glove.6B.200d.txt file from <small><i><a href='https://www.kaggle.com/datasets/incorpes/glove6b200d'>glove.6b.200d.txt'>glove.6B.200d.txt</a></i></small>. Read through the comments at the top of the page in __________ for more details on changing the path names. You can export __________ as a .py file and then run it on the command line by navigating to the directory containing the file and using the command: python3 __________

# Method
## Preprocessing

## Feature Extraction and Vectorization

### Build some tools: Static Word Embeddings with Glove

## Information Extraction / Classification using machine learning

# Data 
## Original Data
We used the LitBank dataset that can be found online at: [LitBank](https://github.com/dbamman/litbank). This dataset contains approximately 2,000 words from 100 English works of fiction (novels and short stories) drawn from the Project Gutenberg public domain found online here: [Project Gutenberg](https://www.gutenberg.org/). The LitBank dataset uses named entities that cover six of the ACE 2005 classes in text:
* People (PER): *Tom Sawyer*, *her daughter*
* Facilities (FAC): *the house*, *the kitchen*
* Geo-political entities (GPE): *London*, *the village*
* Locations (LOC): *the forest*, *the river*
* Vehicles (VEH): *the ship*, *the car*
* Organizations (ORG): *the army*, *the Church*

If a word or symbol is not considered any of the above entities it is marked with an *O*. The 100 English works of fiction can be found in the Data sub-directory in the Original Data ( TSV Format ) folder. Each novel or short story is stored as an individual TSV file.

The original dataset has nested named entities. For example, 

<p align="center">
 <img src="./nested_structure.png" width="720" height="240">
</p>

## Cleaned Data
The "collapseNestedNamedEntities.py" python script that we wrote generated a cleaner version of the LitBank dataset. Specifically we did the following preprocessing tasks to clean the data into the format we wanted for training our model:
* We dealt with these by collapsing the nested layers, selecting the highest non *O* named entity for each token. 
* We appended every sentence from every book into one large file to better be used for training our model.

This clean version of the dataset can be found in the Data sub-directory as "ProcessedLitBankDataset.csv"

We split the data set into three pieces for use in training and evaluation of the model. We use an __ train, __ development or validation, and __ test split.

# Results
## Confusion Matrix

![ConfusionMatrix](./ConfusionMatrix.png "Confusion Matrix")

## Training Versus Validation Loss

![TrainingVsValidationLoss](./TrainingVsValidationLoss.png "Training Vs Validation Loss")

## Precision, Recall, and F1 Performance Measures

![PrecisionRecallandF1](./P_R_F1.png "Precision, Recall, and F1 Performance Measures")

# Discussion

# Future Work

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

The objective of this project is to perform information extraction by applying a trained model to tweets we pull from Twitter. We were allowed to train our model to complete the information extraction task with a dataset of our choice. Information Extraction is a classification problem in which our code will try to determine whether a word or symbol is considered a Named Entity. Named Entities in our dataset cover six of the ACE 2005 categories in text:
* People (PER): *Tom Sawyer*, *her daughter*
* Facilities (FAC): *the house*, *the kitchen*
* Geo-political entities (GPE): *London*, *the village*
* Locations (LOC): *the forest*, *the river*
* Vehicles (VEH): *the ship*, *the car*
* Organizations (ORG): *the army*, *the Church*
If a word or symbol is not considerred any of the above enities it is marked with an *O*. Given any sequence of words or symbols, reffered to as tokens, our model will label each token as one of the named entity classes or as an *O*. Then we must analyze the accuracy of our model. We do this by comparing the model’s predictions to labels made by humans which are assumed to be correct. The model for Information Extraction is a collection of Bidirectional Long Short Term Memory Units ( BiLSTM ) with a Conditional Random Field ( CRF ) layer for Classification. 

The whole process begins by preprocessing a large training dataset. The data used for this project can be found in the Data sub directory. The dataset contains sentences from 100 literary works in which each token has been classified. These sentences must be preprocessed before being vectorized and embedded. After its transformation, the data can then be given to the BiLSTM + CRF model. The model will learn from the data how to predict whether a token is a Named Entity or not and, if so, what named entity it is. We then collect the predictions of our model on a test set of data, and for our results we calculate the precision, recall, F1, and accuracy scores of our predictions.

Each of these steps are explained in further detail in the sections below.

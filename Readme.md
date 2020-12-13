# LDA topic modeling

## setup
The used libraries can be found in requirements.txt and can be installed by running
`pip install -r requirements.txt`

### Overview of files
* The data directory contains some small datasets used for testing. The actual dataset can be downloaded here https://drive.google.com/file/d/1tU6h_jpOCg0Ua9qc8gi6EinthAM670_R/view
This dataset should then be moved to the data directory and should be renamed to `news_dataset.csv`
* The obj directory contains the raw output from our algorithm
* GibbsLDA.py contains our implementation of the LDA model, running this file will the LDA model on the news_dataset.csv` dataset.
* sklearnLDA.py uses the sklearn implementation of the LDA model. we used the result of this script to compare the output from our implementation
* DataPlotter.py contains all the code used to generate the data and tables present in the report
* loadLDA.py loads in the result from GibbsLDA.py and generates the csv file as specified in the assignment.
* top_100_documents_per_topic.csv is the output of loadLDA.py with 20 topics

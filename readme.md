# Approach
## [Solution](./training_nb.ipynb)

*When reading the problem statement the easiest way was to use an OpenAI API key, incorporating langchain lib and passing a prompt with a few examples.*

However this would not make use of the labelled dataset. **On further data exploration I saw that with the 8129 labelled rows, there were only 32 unique categories, which means this problem is two fold**

    - Perform multi-label classification on a given sentence.
    - using the given categories, run sentiment analysis to get whether positive or negative

For the sake of my sanity and ease of readability, I've made two datasets, one to be used for category training and the other to be used for sentiment training. This means we will also be using two models.

The key advantage to this is that we get to locally keep the weights of our pretrained model. So we have reproducable results

Kindly refer to the [notebook](./training_nb.ipynb) for step by step guide 

# Methodology

## Data Preprocessing
  - Break the dataset into two: One for category classification and the other for sentiment analysis training
  - Split the values in each column into seperate tokens. eg. value for money positive -> "value for money" and "positive"
  - Basic preprocessing like dropping all the rows where there is no labelling
  - More info in [preprocess.py](./src/preprocess.py) file
  - Files are [category.csv](./csv_files/category.csv) and [sentiment.csv](./csv_files/sentiment.csv)

## Training
  - Using the BERT transformer for training, specifically 'bert-base-uncased' from BertForSequenceClassification.
  - Using the transformer library which makes the code much more readble and reduces number of lines
  - More information in the notebook
  
## Testing
  - The final block in the python notebook joins both the models and gives us a dictionary output for the input sentence


# NOTE:
I've found that some of the labels are wrong. This has affected training but we still get reasonable answers
eg.
"The two tyres were fitted efficiently and I was unable to attend at the time of the appointment so they fitted me in when I was able to attend. The staff were friendly, generous with their coffee and were able to fit the two tyres on the wheels of my choice. I’m very satisfied!" has 

"garage service negative", "wait time negative", "length of fitting negative"

clearly that is not the case here
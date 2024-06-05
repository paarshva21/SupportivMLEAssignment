import pandas as pd
import re
from datasets import Dataset
from transformers import T5Tokenizer
import warnings
warnings.filterwarnings("ignore")

# Helper function to only keep text in the questions and answers 
# and to remove any other unnecessary characters that may be there
def text_transform(df):
  # pattern for removing unnecessary characters
  pattern = r"(\\|\s\s+|\n|\r|\t|\f|\v)"
  # Dataframe.applymap() method applies a function that accepts and returns a scalar to every element of a DataFrame
  return df.applymap(lambda x: re.sub(pattern, "", x))

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# loading the dataset given for the task
medical_qna_df = pd.read_csv('intern_screening_dataset.csv')

# we check for any potential rows having Nan values
medical_qna_df.isna().sum()

# we drop the rows having Nan values
medical_qna_df.dropna(inplace = True)

# we drop any potential duplicate rows
medical_qna_df.drop_duplicates(inplace = True)

# shuffling the dataset whilst keeping the indexing intact
medical_qna_df = medical_qna_df.sample(frac = 1).reset_index(drop = True)

# to only keep text in the dataset
medical_qna_df = text_transform(medical_qna_df)

# The final dataset is saved
medical_qna_df.to_csv('final.csv')

dataset = Dataset.from_pandas(medical_qna_df[:5000])
dataset = dataset.train_test_split(test_size = 0.3)

# Define our preprocessing function
def preprocess_function(dataset):
    # The "inputs" are the tokenized answer:
    inputs = [q for q in dataset["question"]] 
    model_inputs = tokenizer(inputs, max_length = 256, truncation = True)
    
    # The "labels" are the tokenized outputs:
    labels = tokenizer(text_target = dataset["answer"], max_length = 512, truncation = True) 
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Map the preprocessing function across our dataset then save it
dataset = dataset.map(preprocess_function, batched = True)
dataset.save_to_disk("dataset") 
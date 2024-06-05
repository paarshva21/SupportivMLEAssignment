import numpy as np
import nltk
from datasets import Dataset
import evaluate
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Function to compute metrics for evaluation


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Replace label ids that are -100 with the tokenizer's pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode the predicted and actual labels to text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Ensure ROUGE metric has newlines after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    # Compute the ROUGE metric
    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels, use_stemmer=True)
    return result


# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# Create a data collator that will dynamically pad the inputs and labels
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define training arguments for the Seq2Seq model
training_args = Seq2SeqTrainingArguments(
    output_dir="QnAModel",               # Output directory
    eval_strategy="epoch",               # Evaluation strategy
    learning_rate=3e-4,                  # Learning rate
    per_device_train_batch_size=4,       # Batch size for training
    per_device_eval_batch_size=2,        # Batch size for evaluation
    weight_decay=0.01,                   # Weight decay for regularization
    save_total_limit=3,                  # Limit the total number of checkpoints saved
    num_train_epochs=2,                  # Number of training epochs
    predict_with_generate=True,          # Use generation during prediction
    push_to_hub=False                    # Do not push the model to the hub
)

# Download NLTK punkt tokenizer
nltk.download("punkt", quiet=True)
# Load the ROUGE metric
metric = evaluate.load("rouge")

# Load training and test datasets
dataset_test = Dataset.from_file('dataset/test/data-00000-of-00001.arrow')
dataset_train = Dataset.from_file('dataset/train/data-00000-of-00001.arrow')

# Set up the trainer with model, arguments, datasets, tokenizer, data collator, and metrics function
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the improved model
trainer.save_model("modelSmallImproved")

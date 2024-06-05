from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings("ignore")

# Define the maximum length for the generated answers
max_length = 92

# Load the custom pre-trained models
model = T5ForConditionalGeneration.from_pretrained("model")
modelBase = T5ForConditionalGeneration.from_pretrained("modelBase")

# Load the tokenizer from the Google FLAN-T5 small and base models
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
tokenizerBase = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Create pipelines for text-to-text generation using the models and tokenizers
pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)
pipeBase = pipeline(task="text2text-generation",
                    model=modelBase, tokenizer=tokenizerBase)

# Define the prompt for the questions
prompt = "Please answer this question: "

# List of questions to be answered
question1 = "What is glaucoma?"
question2 = "Is mineral deficiency harmful?"
question3 = "What causes Urine Blockage in Newborns?"

# Process and print answers for the first question
print("\nQuestion:", question1)
print("Small model answer:", pipe(prompt + question1,
      max_length=max_length)[0]['generated_text'], "\n")
print("Base model answer:", pipeBase(prompt + question1,
      max_length=max_length)[0]['generated_text'], "\n")

# Process and print answers for the second question
print("Question:", question2)
print("Small model answer:", pipe(prompt + question2,
      max_length=max_length)[0]['generated_text'], "\n")
print("Base model answer:", pipeBase(prompt + question2,
      max_length=max_length)[0]['generated_text'], "\n")

# Process and print answers for the third question
print("Question:", question3)
print("Small model answer:", pipe(prompt + question3,
      max_length=max_length)[0]['generated_text'], "\n")
print("Base model answer:", pipeBase(prompt + question3,
      max_length=max_length)[0]['generated_text'], "\n")

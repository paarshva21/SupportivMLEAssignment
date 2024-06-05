# Supportiv MLE Assignment

## Assumptions

- Since the dataset only contains two columns (Question and Answer), I have assumed that there is no context or reference text from where the answer is extracted or inferred from in the dataset.
- I have assumed that the user will input a question and get a single answer as output, instead of there being multiple answers being presented to the person passing the query as input to the model.

## Information Gathering

### There are many ways to approach a question answering task. The three that I explored in my attempt to do this task were:
1. Converting the input question to vector format via TF-IDF vectorizer. Then, after comparing the input question with all the other questions in the dataset with cosine similarity, the answer of the question with the highest similarity is shown. However, this is obviously a very primitive, simple and shortsighted approach that will not actually generate any new answers if required. Not to mention that an answer will always be fetched for totally unrelated input questions as well. 
2. Finetuning a BERT (Bidirectional Encoder Representations from Transformers) model for the task. The powerful BERT architecture utilises attention mechanism to factor in context, which makes them extremely powerful. This means that finetuning BERT for question answering purposes requires there to be a reference text or context for each answer to a question, which is absent in our dataset. It could be manufactured by appending the answer to the question and creating a sort of "reference" from which the answer is inferred from, but it did not seem like a sound approach to me as I am basically tampering with the dataset to assume context that simply for the purposes of enforcing BERT.
3. Finetuning a sequence to sequence text generator model like T5 on the given dataset. This made the most sense to me, as our dataset was such that it contained only questions and answers. Hence, the task was essentially reduced to generation a sequence (a.k.a the answer) from a given input question.

### So, I chose to move on with the third approach.

## Model

### I finetuned the small and base variants of Flan T5 model for our given question answering task. The Flan T5 model is an enhanced version of T5 that has been finetuned in a mixture of tasks. The original T5 is an encoder-decoder model that converts all NLP problems into a text-to-text format. It is trained using teacher forcing. This means that for training, we always need an input sequence and a corresponding target sequence. 

<p align="center">
         <img width="800" src="https://github.com/paarshva21/SupportiveMLEAssignment/assets/113699487/2c6d99bf-e878-40bc-80f6-e52634e3af78">
</p>


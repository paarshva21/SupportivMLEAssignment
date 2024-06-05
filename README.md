# Supportiv MLE Assignment

## Assumptions

- Since the dataset only contains two columns (Question and Answer), I have assumed that there is no context or reference text from where the answer is extracted or inferred from in the dataset.
- I have assumed that the user will input a question and get a single answer as output, instead of there being multiple answers being presented to the person passing the query as input to the model.

## Information Gathering

There are many ways to approach a question answering task. The three that I explored in my attempt to do this task were:

1. Converting the input question to vector format via TF-IDF vectorizer. Then, after comparing the input question with all the other questions in the dataset with cosine similarity, the answer of the question with the highest similarity is shown. However, this is obviously a very primitive, simple and shortsighted approach that will not actually generate any new answers if required. Not to mention that an answer will always be fetched for totally unrelated input questions as well. 
2. Finetuning a BERT (Bidirectional Encoder Representations from Transformers) model for the task. The powerful BERT architecture utilises attention mechanism to factor in context, which makes them extremely powerful. This means that finetuning BERT for question answering purposes requires there to be a reference text or context for each answer to a question, which is absent in our dataset. It could be manufactured by appending the answer to the question and creating a sort of "reference" from which the answer is inferred from, but it did not seem like a sound approach to me as I am basically tampering with the dataset to assume context that simply for the purposes of enforcing BERT.
3. Finetuning a sequence to sequence text generator model like T5 on the given dataset. This made the most sense to me, as our dataset was such that it contained only questions and answers. Hence, the task was essentially reduced to generation a sequence (a.k.a the answer) from a given input question.

So, I chose to move on with the third approach.

## Model

The Flan T5 model is an enhanced version of T5 that has been finetuned in a mixture of tasks. 

The original T5 model is an encoder-decoder model that converts all NLP problems into a text-to-text format. It is trained using teacher forcing. This means that for training, we always need an input sequence and a corresponding target sequence. 

<p align="center">
         <img width="800" src="https://github.com/paarshva21/SupportiveMLEAssignment/assets/113699487/2c6d99bf-e878-40bc-80f6-e52634e3af78">
</p>

## Training

I finetuned the small and base variants of Flan T5 model for our given question answering task. Their respective links for further reading:

- [Flan T5 Small](https://huggingface.co/google/flan-t5-small)
- [Flan T5 Base](https://huggingface.co/google/flan-t5-base)
  
They were finetuned on 2500 question answer samples from the shuffled dataset, as my laptop hardware is not the best and to save on training time. These 2500 samples were split into train and test datasets in 70:30 ratio.

To evaluate NLP models, a metric known as ROUGE is often used. It works by emphasizing a recall-oriented assessment. This means it focuses on how much of the important content from the human-written summary is captured in the machine-generated summary. To achieve this, ROUGE examines various sets of words called n-grams, which are just word groups. For example,ROUGE-1, for instance, looks at individual words or unigrams, while ROUGE-2 considers pairs of words or bigrams, and so on. Additionally, ROUGE-L examines the longest common subsequence between the machine-generated and human reference summaries.

The ROUGE-Lsum is related to the ROUGE-L metric but applies a slightly different calculation method. It applies the ROUGE-L calculation method at the sentence level and then aggregates all the results for the final score. This metric is seen as more suitable for tasks where sentence level extraction is valuable such as extractive summarization tasks.
In simpler terms, whereas ROUGE-L looks at the summary as a whole, ROUGE-Lsum considers sentence-level information, potentially providing more granularity in some use cases.

ROUGE-L ignores newlines and computes the LCS for the entire text. ROUGE-Lsum splits the text into sentences based on newlines and computes the LCS for each pair of sentences and take the average score for all sentences.

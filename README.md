# Supportiv MLE Assignment

## Assumptions

- Since the dataset only contains two columns (Question and Answer), I have assumed that there is no context or reference text from where the answer is extracted or inferred from in the dataset.
- I have assumed that the user will input a question and get a single answer as output, instead of there being multiple answers being presented to the person passing the query as input to the model.
- Since the assignment asked us to not use any external LLM APIs, I have also refrained from finetuning models like Llama-2 (using LoRA) or GPT-2. I have attempted to create the question answering model by following a more pure NLP driven approach. 

## Information Gathering

There are many ways to approach a question answering task. The three that I explored in my attempt to do this task were:

1. Converting the input question to vector format via TF-IDF vectorizer. Then, after comparing the input question with all the other questions in the dataset with cosine similarity, the answer of the question with the highest similarity is shown. However, this is obviously a very primitive, simple and shortsighted approach that will not actually generate any new answers if required. Not to mention that an answer will always be fetched for totally unrelated input questions as well. 
2. Finetuning a BERT (Bidirectional Encoder Representations from Transformers) model for the task. The BERT architecture utilises attention mechanism to factor in context, which makes them extremely powerful. This means that finetuning BERT for question answering purposes requires there to be a reference text or context for each answer to a question, which is absent in our dataset. It could be manufactured by appending the answer to the question and creating a sort of "reference" from which the answer is inferred from, but it did not seem like a sound approach to me as I am basically tampering with the dataset to assume context; that too simply for the purposes of forcefully enforcing BERT.
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
  
They were finetuned on 2500 question answer samples from the shuffled dataset for 2 epochs, as my laptop hardware is not the best and to save on training time. These 2500 samples were split into train and test datasets in 70:30 ratio.

To evaluate NLP models, a metric known as ROUGE is often used. It works by emphasizing a recall-oriented assessment. This means it focuses on how much of the important content from the human-written summary is captured in the machine-generated summary. To achieve this, ROUGE examines various sets of words called n-grams, which are just word groups. For example,ROUGE-1, for instance, looks at individual words or unigrams, while ROUGE-2 considers pairs of words or bigrams, and so on. Additionally, ROUGE-L examines the longest common subsequence between the machine-generated and human reference summaries.

The ROUGE-Lsum is related to the ROUGE-L metric but applies a slightly different calculation method. It applies the ROUGE-L calculation method at the sentence level and then aggregates all the results for the final score. This metric is seen as more suitable for tasks where sentence level extraction is valuable such as extractive summarization tasks.
In simpler terms, whereas ROUGE-L looks at the summary as a whole, ROUGE-Lsum considers sentence-level information, potentially providing more granularity in some use cases.

ROUGE-L ignores newlines and computes the LCS for the entire text. ROUGE-Lsum splits the text into sentences based on newlines and computes the LCS for each pair of sentences and take the average score for all sentences.

Below are the metrics I used and how they performed for the small model:

- eval_loss: This metric measures how well the model performed on a validation set. Lower scores indicate better performance. The values range from 2.09 to 2.14.
- eval_rouge1, eval_rouge2, eval_rougeL, eval_rougelsum: These metrics are used for text summarization tasks and measure how well the model's generated summaries match human-written summaries. Higher scores indicate better performance. The values range from 0.14 to 0.16 for ROUGE-1, 0.09 to 0.10 for ROUGE-2, 0.14 to 0.15 for ROUGE-L, and 0.15 for ROUGE-Lsum.
- eval_runtime, eval_samples_per_second, eval_steps_per_second: These metrics measure how long it took to evaluate the model on the validation set. Lower scores indicate faster evaluation. The values range from 426 seconds to 1186 seconds for eval_runtime, 1.26 to 3.52 samples per second for eval_samples_per_second, and 0.31 to 0.88 steps per second for eval_steps_per_second.
- train_loss: This metric measures how well the model performed on the training set. Lower scores indicate better performance. The value is 2.44.
- train_runtime, train_samples_per_second, train_steps_per_second: These metrics measure how long it took to train the model on the training set. Lower scores indicate faster training. The values are 5474 seconds for train_runtime, 1.28 samples per second for train_samples_per_second, and 0.16 steps per second for train_steps_per_second.
- epoch: This metric indicates the number of times the model has been trained on the entire dataset. The values range from 1.0 to 2.0.

And, for the base model:

- eval_loss: This metric measures how well the model performed on a validation set. Lower scores indicate better performance. The values range from 1.93 to 2.04.
- eval_rouge1, eval_rouge2, eval_rougeL, eval_rougelsum: These metrics are used for text summarization tasks and measure how well the model's generated summaries match human-written summaries. Higher scores indicate better performance. The values range from 0.13 to 0.14 for ROUGE-1, 0.08 to 0.08 for ROUGE-2, 0.12 to 0.13 for ROUGE-L, and 0.13 to 0.14 for ROUGE-Lsum.
- eval_runtime, eval_samples_per_second, eval_steps_per_second: These metrics measure how long it took to evaluate the model on the validation set. Lower scores indicate faster evaluation. The values range from 331 seconds to 356 seconds for eval_runtime, 0.84 to 0.98 samples per second for eval_samples_per_second, and 0.42 to 0.45 steps per second for eval_steps_per_second.
- train_loss: This metric measures how well the model performed on the training set. Lower scores indicate better performance. The value is 2.05.
- train_runtime, train_samples_per_second, train_steps_per_second: These metrics measure how long it took to train the model on the training set. Lower scores indicate faster training. The values are 4852 seconds for train_runtime, 0.29 samples per second for train_samples_per_second, and 0.07 steps per second for train_steps_per_second.
- epoch: This metric indicates the number of times the model has been trained on the entire dataset. The values are 2.0.

## Potential Improvements

The model ROUGE scores are very low, which is to be expected as the training approach and environment is very computationally light. Also, the answers are very restricted to the dataset, as is to be expected from sequence to sequence models. But if finetuned on the entire dataset, then the question answering model will work extremely well, with respect to the answers in the dataset.

## Sample Model Answers

1. Question: What is glaucoma?
   
   Small model answer: Glaucoma is a rare disease that affects the eyes and eyes. It is caused by a combination of factors, including the presence of a glaucoma, and the presence of a glaucoma. The symptoms of 
   glaucoma vary widely, and may include a glaucoma that is a swollen eyelid, a swollen eyelid 

   Base model answer: Glaucoma is a condition that causes a stenting of the eye, which is a symptom of glaucoma. Glaucoma is a condition that affects the eyes, ears, and nose. Glaucoma is caused by mutations in 
   the X-linked gene, which means that the X-linked gene is involved in the development of the glaucoma.

2. Question: Is mineral deficiency harmful?
   
   Small model answer: The mineral deficiency is a condition that affects the body's ability to absorb minerals. The mineral deficiency is a condition that affects the body's ability to absorb minerals. The 
   mineral deficiency is a condition that affects the body's ability to absorb minerals. The mineral deficiency is a condition that affects the body's ability to absorb minerals.

   Base model answer: The risk of developing mineral deficiency is high. The risk of developing mineral deficiency is high. The risk of developing mineral deficiency is high.

3. Question: What causes Urine Blockage in Newborns?
   
   Small model answer: The urethra is a bacterial infection that causes a bacterial infection. The urethra is a bacterial infection that causes a bacterial infection. The urethra is a bacterial infection that 
   causes a bacterial infection. The urethra is a bacterial infection that causes a bacterial infection.

   Base model answer: Urine blockage is a condition that occurs in newborns. It is caused by a mutation in the ER1 gene, which provides instructions for making urine. The ER1 gene provides instructions for making 
   urine, but it is not known how it causes the condition. The ER1 gene provides instructions for making urine, but it is not known how it causes the condition.

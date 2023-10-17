# Sentiment Analysis on IMDb Movie Reviews

## Intro
Sentiment analysis is a use case of text classification which consists of assigning a category to a given text. It's a powerful Natural Language Processing (NLP) technique that makes it possible to automatically analyze what people think about a certain topic. This can help companies and individuals to quickly make more informed decisions. Sentiment analysis has for example applications in [social media, customer service and market research](https://www.taus.net/resources/blog/what-is-sentiment-analysis-types-and-use-cases).  

Have you ever wanted to summarize all the reviews for a given movie to decide if that movie is worth watching without wasting time reading all the reviews?
The aim of this project is to build a model that can accurately predict whether a movie review is positive or not.  The model will be made available to interact with through a simple Streamlit App.

## Data
The dataset used to develop our binary sentiment classifier is an [IMDB dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) which consists of about 50K movie reviews.  
There was however a small proportion of duplicates (0.8%) which I discarded. So the final total number of samples was 49582.  
The number of positive and negative reviews is well balanced.
## Model and Training
I leveraged the pretrained BERT (Bidirectional Encoder Representations from Transformers) made available by Hugging Face. The model (bert-base-uncased) was fine-tuned using the following hyper-parameters:
* `Learning rate = 2e-5` using `AdamW` optimizer
* `Linear scheduler` with `num_warmup_steps = 0`
* `Maximum sequence length = 128`
* `Batch size = 32`
* `Number of training epochs = 5`

[Here](https://huggingface.co/docs/transformers/training) is a nice tutorial from Hugging Face explaining how to fine-tune a pretrained model.

The model is quite heavy (approx. 427 MB). So after fine-tuning, I pushed it to the [Hugging Face hub](https://huggingface.co/MLphile/fine_tuned_bert-movie_review), from where it can be accessed using the following checkpoint address: `MLphile/fine_tuned_bert-movie_review`.
## Evaluation
The accuracy on the validation set reached 89.35. Evaluation results on the test set can be seen in the table below.
| class | precision | recall | f1-score | support |
| --- | --- | --- |--- |--- |
| 0 | 0.91 | 0.89 | 0.90 | 3705 |
| 1 | 0.90 | 0.92 | 0.91 | 3733 |

## How to run
The streamlit app is available [here](https://mlphile-bert-on-movie-reviews-srcapp-9n3r3m.streamlit.app/). At the beginning, it takes few secondes to load the model and its tokenizer. Have fun playing with it :smiley:.  
However, if you want to run it locally or want to try out different things, you can clone this repo and preferably create a virtual environment to install all dependencies (see requirements.txt).  
* `cd` into the project folder
* `streamlit run scr/app.py`
and voila you have it running in your browser!

If you face any issues or want to ask something feel free to get in touch 😉.

## References
* [What is Sentiment Analysis? Types and Use Cases](https://www.taus.net/resources/blog/what-is-sentiment-analysis-types-and-use-cases)
* [Hugging Face](https://huggingface.co/docs/transformers/index)

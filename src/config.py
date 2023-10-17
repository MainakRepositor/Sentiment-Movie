DATA_PATH = '../data/IMDB_Dataset.csv'
SAVED_MODEL_PATH = '../model'
BERT_CHECKPOINT = 'bert-base-uncased'
FINETUNED_CHECKPOINT = 'MLphile/fine_tuned_bert-movie_review'

MAPPING = {0:'negative', 1:'positive'}

MAX_LEN = 128
BATCH_SIZE = 32
NUM_CLASSES = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS= 5
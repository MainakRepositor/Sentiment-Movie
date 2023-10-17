import torch
from torch.utils.data import Dataset
import re


def clean_text(text):
    """Removes extra whitespaces and html tags from text."""
    # remove weird spaces
    text =  " ".join(text.split())
    # remove html tags
    text = re.sub(r'<.*?>', '', text)
    return text


# Class for custom dataset
class CustomDataset(Dataset):
    def __init__(self, review, target, tokenizer, max_len, clean_text=None):
        self.clean_text = clean_text
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        y = torch.tensor(self.target[idx], dtype=torch.long)
        X = str(self.review[idx])
        if self.clean_text:
            X = self.clean_text(X)
        
        encoded_X = self.tokenizer(
            X, 
            return_tensors = 'pt', 
            max_length = self.max_len, 
            truncation=True,
            padding = 'max_length'
            )

        return {'input_ids': encoded_X['input_ids'].squeeze(),
                'attention_mask': encoded_X['attention_mask'].squeeze(),
                'labels': y}




############################### Training, validation, testing ##############################
# Traing loop for one epoch
def train_epoch(model, dataloader, optimizer, scheduler, device, progress_bar):

    losses = []
    accuracies = []

    model.train()
    for batch in dataloader:

        optimizer.zero_grad()
        batch = {k:v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1)
        acc = torch.sum(preds == batch['labels']) / len(preds)
        accuracies.append(acc)
        losses.append(loss)

        progress_bar.update(1)
    
    return torch.tensor(losses, dtype=torch.float).mean().item(), torch.tensor(accuracies).mean().item()


# Evaluation loop for one epoch
def eval_epoch(model, dataloader, device):
    losses = []
    accuracies = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

            batch = {k:v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            preds = torch.argmax(outputs.logits, dim=1)
            acc = torch.sum(preds == batch['labels']) / len(preds)
            accuracies.append(acc)
            losses.append(loss)
        
        return torch.tensor(losses, dtype=torch.float).mean().item(), torch.tensor(accuracies).mean().item()
    
    


# For final evaluation on test set
def test(model, dataloader, device):
    y_preds = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

            batch = {k:v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
        
         
            y_preds.extend( torch.argmax(outputs.logits, dim=1) )
            y_true.extend( batch['labels'])
            
        return y_preds, y_true




################################# Prediction ########################################

def preprocess(text,tokenizer, max_len, clean_text=None):
    """Tokenizes the text. If a clean_text function is provided,
    then the cleaning process is carried out before tokenization.

    Args:
        text (str): The text to preprocess.
        tokenizer (Hugging Face Tokenizer): Tokenizer coming along with a given model.
        max_len (int): Maximum sequence length, taken as reference for truncation
                        and padding during the tokenization process.
        clean_text (function): For prior text cleaning, default to None.

    Returns:
        dict: A dictionary containing input ids and the corresponding attention mask.
    """
    text = str(text)
    if clean_text:
        text = clean_text(text)
    tokenized_text = tokenizer(
            text, 
            return_tensors = 'pt', 
            max_length = max_len, 
            truncation=True,
            padding = 'max_length'
            )

    return {
            'input_ids':tokenized_text['input_ids'],
            'attention_mask':tokenized_text['attention_mask']
            }

def classify(inputs, model, mapping):
    """ Predicts the sentiment of text.

    Args:
        inputs (dict): A dictionary containing input ids and the corresponding attention mask
        model (Torch model): Model to carry out prediction.
        mapping (dict): A mapping from model outputs as keys (int) to human-readable labels as values (str).

    Returns:
        dict: Contains the predicted text label (str) and the corresponding probability (float).
    """
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs).logits
        probabilities = torch.softmax(outputs, dim=1)
        label = torch.argmax(probabilities, dim=1).item()

    return {
        'Label': mapping[label],
        'Confidence': round(probabilities.squeeze()[label].item(), 2)
    }
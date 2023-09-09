import data_loader as datalib
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel


file_path  = 'Data\IMDB Dataset.csv'
data = datalib.load_data(file_path)

X = data['review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_Dataset = datalib.create_dataset(texts = X_train, labels = y_train, tokenizer = tokenizer, max_length = 50)
test_Dataset = datalib.create_dataset(texts = X_test, labels = y_test, tokenizer = tokenizer, max_length = 50) 

breakpoint()
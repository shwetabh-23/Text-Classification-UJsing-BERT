import data_loader as datalib
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

file_path  = 'Data\IMDB Dataset.csv'
X, y = datalib.load_data(file_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = datalib.create_dataset(texts = X_train, labels = y_train, tokenizer = tokenizer, max_length = 128)
test_dataset = datalib.create_dataset(texts = X_test, labels = y_test, tokenizer = tokenizer, max_length = 128) 

train_loader = datalib.data_loader(train_dataset, batch_size= 25)
test_loader = datalib.data_loader(test_dataset, batch_size= 25)
# to visualize data
#datalib.visualize_distribution(train_dataset)

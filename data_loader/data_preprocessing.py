

def preprocess(data):
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return data
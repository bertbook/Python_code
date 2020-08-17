from deeppavlov import build_model, configs

def build_sentiment_model():
    model = build_model(configs.classifiers.insults_kaggle_bert, download=True)
    return model


test_input = ['hey, how are you?', 'You are so dumb!']
sentiment_model = build_sentiment_model()
results = sentiment_model(test_input)
print(results)
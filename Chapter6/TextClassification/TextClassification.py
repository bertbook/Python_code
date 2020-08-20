from sklearn.datasets import fetch_20newsgroups
import ktrain
from ktrain import text

def download_dataset():
    categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
    train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    train_classes = train.target_names
    x_train, y_train = train.data, train.target
    x_test, y_test = test.data, test.target
    return x_train, y_train, x_test, y_test, train_classes

def create_text_classification_model():
    MODEL_NAME = 'distilbert-base-uncased'
    x_train, y_train, x_test, y_test, train_classes = download_dataset()
    t = text.Transformer(MODEL_NAME, maxlen=500, classes=train_classes)
    trn = t.preprocess_train(x_train, y_train)
    val = t.preprocess_test(x_test, y_test)
    model = t.get_classifier()
    classification_model = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
    classification_model.fit_onecycle(5e-5, 4)
    return classification_model, t

def predict_category(classification_model, t, input_text):
    predictor = ktrain.get_predictor(classification_model.model, preproc=t)
    results = predictor.predict(input_text)
    return results

""" 
classification_model, t = create_text_classification_model()
input_text = 'Babies with down syndrome have an extra chromosome.'
print(predict_category(classification_model, t, input_text))
"""

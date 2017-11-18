import numpy as np
import math
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from pandas import DataFrame
from utils.database import ProductManager


class TextClassifier:
    def __init__(self):
        self.text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 4))),
                             ('tfidf', TfidfTransformer(use_idf=True)),
                             ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=200, n_jobs=-1))])
        self.pred_list = []
        self.text_prediction = []
        self.accuracy = 0

        self.test_data = []
        self.test_label = []
        self.train_data = []
        self.train_label = []

    def read_data(self, new_data=False):
        if os.path.exists('train_data.pickle') and new_data is False:
            with open('train_data.pickle', 'rb') as f:
                self.train_data = pickle.load(f)
            with open('train_label.pickle', 'rb') as f:
                self.train_label = pickle.load(f)

            with open('test_data.pickle', 'rb') as f:
                self.test_data = pickle.load(f)
            with open('test_label.pickle', 'rb') as f:
                self.test_label = pickle.load(f)
        else:
            product_db = ProductManager()
            inserted_cate_list = [cate for cate in product_db.retrieve_products_by_cate()]
            for prod in inserted_cate_list[len(inserted_cate_list)//10:]:
                self.train_data.append(prod.get('product_name'))
                self.train_label.append(prod.get('product_cate'))
            with open('train_data.pickle', 'wb') as f:
                pickle.dump(self.train_data, f)

            with open('train_label.pickle', 'wb') as f:
                pickle.dump(self.train_label, f)

            for prod in inserted_cate_list[:len(inserted_cate_list)//10]:
                self.test_data.append(prod.get('product_name'))
                self.test_label.append(prod.get('product_cate'))

            with open('test_data.pickle', 'wb') as f:
                pickle.dump(self.test_data, f)

            with open('test_label.pickle', 'wb') as f:
                pickle.dump(self.test_label, f)

    def fit(self, load_model=False):
        filename = 'text_svm_model.sav'
        if load_model:
            self.text_clf = joblib.load(filename)
        else:
            assert self.train_data and self.train_label
            self.text_clf.fit(self.train_data, self.train_label)
            # save the model to disk
            joblib.dump(self.text_clf, filename)

    def evaluate(self):
        scores, classes = self.text_clf.predict_prob(self.test_data)
        classes = [int(cls) for cls in classes]
        scores_list = []
        for score in scores:
            tmp = DataFrame({'score': score, 'classes': classes})
            tmp = tmp.sort_values(by='classes')
            scores_list.append([sc[1] for sc in tmp.values])

        self.text_prediction = [self.softmax(score) for score in scores_list]
        predicted = np.array(self.text_clf.predict(self.test_data))
        self.accuracy = np.mean(predicted == self.test_label)
        self.pred_list.append(np.mean(predicted == self.test_label))
        self.test_label = [int(cls) - 1 for cls in self.test_label]
        return self.accuracy

    def predict(self, sentence):
        scores, classes = self.text_clf.predict_prob(sentence)
        classes = [int(cls) for cls in classes]
        tmp = DataFrame({'score': scores[0], 'classes': classes})
        tmp = tmp.sort_values(by='classes')
        return [sc[1] for sc in tmp.values]

    def softmax(self, X):
        return [round(i / sum([math.exp(i) for i in X]), 3) for i in [math.exp(i) for i in X]]

if __name__ == '__main__':
    # text_classifier 초기화
    text_classifier = TextClassifier()

    # ---------------------------------------------------------------------------------------------
    # db로부터 랜덤하게 모든 상품을 가져와서 앞에 10%를 test data로 뒤에 90%를 train data로 pickle로 저장
    # 이미 만들어 놓은게 있다면 load
    text_classifier.read_data(new_data=False)

    # train data 학습
    text_classifier.fit(load_model=True)

    # Evaluate
    accuracy = text_classifier.evaluate()
    print('Accuracy : ', accuracy)  # 0.81

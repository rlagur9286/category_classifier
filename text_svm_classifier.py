import numpy as np
import math
import pickle
import os

from konlpy.tag import Twitter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from pandas import DataFrame
from utils.database import ProductManager
from utils.database import CategoryManager


class MyPipeline(Pipeline):
    def predict_prob(self, X):
        """
        각 클래스별 스코어 리턴 하는 함수 (sklearn에 없어서 추가한 코드)
        :param X: X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        :return: 
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        scores = self.steps[-1][-1].decision_function(Xt)
        return scores, self.steps[-1][-1].classes_


class TextClassifier:
    def __init__(self):
        self.text_clf = MyPipeline([('vect', CountVectorizer(ngram_range=(1, 4))),
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
        """
        DB로부터 저장된 데이터를 읽어와서 편하게 불러서 쓸수 있도록 pickle 형태로 저장 
        :param new_data: new_data가 True이면 기존에 저장한 데이터가 있어도 DB로부터 다시 읽어옴
        :return: 
        """

        # 이미 저장된 pickle 파일이 있고 newe_data flag가 False라면 기존 pickle 데이터 로드
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
            # retrieve 할 때 특정 cate 인자가 없으면 products를 랜덤하게 전부 가져옴
            product_db = ProductManager()
            inserted_products = [prod for prod in product_db.retrieve_products_by_cate()]
            print("{}개의 상품 정보 로드".format(len(inserted_products)))

            # 랜덤하게 가져온 데이터 중 뒤에 90%를 train datset으로
            for prod in inserted_products[len(inserted_products)//10:]:
                self.train_data.append(tokenize(prod.get('product_name')))
                self.train_label.append(prod.get('category_seq'))

            # 다음부터는 쉽게 가져올 수 있도록 pickle 덤프
            with open('train_data.pickle', 'wb') as f:
                pickle.dump(self.train_data, f)
            with open('train_label.pickle', 'wb') as f:
                pickle.dump(self.train_label, f)

            # 랜덤하게 가져온 데이터 중 앞에 10%를 test dataset으로
            for prod in inserted_products[:len(inserted_products)//10]:
                self.test_data.append(tokenize(prod.get('product_name')))
                self.test_label.append(prod.get('category_seq'))

            # 다음부터는 쉽게 가져올 수 있도록 pickle 덤프
            with open('test_data.pickle', 'wb') as f:
                pickle.dump(self.test_data, f)
            with open('test_label.pickle', 'wb') as f:
                pickle.dump(self.test_label, f)

    def fit(self, load_model=False):
        """
        학습 데이터를 가지고 모델을 학습하는 코드
        :param load_model: load_model이 True라면 새롭게 학습 시키지 않고 기존의 모델을 가져옴
        :return: 
        """

        filename = 'text_svm_model.sav'
        if load_model:
            self.text_clf = joblib.load(filename)
        else:
            # 학습 데이터가 없다면 assert
            assert self.train_data and self.train_label
            # 모델을 학습
            self.text_clf.fit(self.train_data, self.train_label)
            # save the model to disk
            joblib.dump(self.text_clf, filename)

    def evaluate(self):
        """
        학습한 모델을 test 데이터로 Evaluate 하는 함수
        :return: test 데이터의 정확도 
        """
        scores, classes = self.text_clf.predict_prob(self.test_data)
        classes = [int(cls) for cls in classes]
        scores_list = []
        for score in scores:
            tmp = DataFrame({'score': score, 'classes': classes})
            tmp = tmp.sort_values(by='classes')
            scores_list.append([sc[1] for sc in tmp.values])

        self.text_prediction = [softmax(score) for score in scores_list]
        predicted = np.array(self.text_clf.predict(self.test_data))
        self.accuracy = np.mean(predicted == self.test_label)
        self.pred_list.append(np.mean(predicted == self.test_label))
        self.test_label = [int(cls) - 1 for cls in self.test_label]
        return self.accuracy

    def predict(self, sentence):
        tokened_sentence = tokenize(sentence)
        scores, classes = self.text_clf.predict_prob([tokened_sentence])
        classes = [int(cls) for cls in classes]
        tmp = DataFrame({'score': scores[0], 'classes': classes})
        tmp = tmp.sort_values(by='classes')
        return [sc[1] for sc in tmp.values]


def softmax(X):
    return [round(i / sum([math.exp(i) for i in X]), 3) for i in [math.exp(i) for i in X]]


def tokenize(doc):
    pos_tagger = Twitter()
    result = []
    for t in pos_tagger.pos(doc, norm=True):
        if t[1] not in ['Punctuation']:
            result.append('/'.join(t))
    return ' '.join(result)

if __name__ == '__main__':
    # text_classifier 초기화
    text_classifier = TextClassifier()

    # ---------------------------------------------------------------------------------------------
    # db로부터 랜덤하게 모든 상품을 가져와서 앞에 10%를 test data로 뒤에 90%를 train data로 pickle로 저장
    # 이미 만들어 놓은게 있다면 load
    text_classifier.read_data(new_data=False)

    # train data 학습
    text_classifier.fit(load_model=False)

    # Evaluate
    accuracy = text_classifier.evaluate()
    print('Accuracy : ', accuracy)

    # Predict
    text_prediction = text_classifier.predict(sentence='아버지가 가방에 들어가신다')    # 0.76

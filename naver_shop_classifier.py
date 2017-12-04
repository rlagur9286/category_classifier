import tensorflow as tf
import numpy as np
import os
import time

from PIL import Image
from pandas import DataFrame
from text_cnn_classifier import TextCNNClassifier
from text_svm_classifier import softmax
from text_cnn_classifier import tokenize
from img_classifier import ImageClassifier
from urllib import request
from utils.database import CategoryManager


class Classifier:
    """
    Image와 Text Classifier 결과를 앙상블을 통해 최종 Classify
    """
    def __init__(self):
        self.text_classifier = TextCNNClassifier()
        self.image_classifier = ImageClassifier()
        self.image_prediction = []
        self.text_prediction = []
        self.label = []

    def evaluate(self):
        """
        Test set을 image와 text 모두를 가지고 evaluate 한 결과
        :return: 
        """

        # Test dataset 모두를 가져옴
        with open('data_/test.txt', 'r', encoding='utf-8') as f:
            test_data = f.readlines()

        # 모든 test data 에 대해서...
        for t_data in test_data:
            try:
                # image data를 가지고 score를 구함
                product_seq = t_data.split(',')[2].strip()
                category_seq = t_data.split(',')[1].strip()
                self.label.append(int(category_seq) - 1)
                file_name = str(product_seq) + '.jpg'
                img_path = 'img/test/' + str(category_seq) + '/' + file_name
                image = tf.gfile.FastGFile(img_path, 'rb').read()
                prediction = self.image_classifier.sess.run(self.image_classifier.logits, {'DecodeJpeg/contents:0': image})[0]
                tmp = {}
                for pred, cls in zip(prediction, self.image_classifier.labels):
                    tmp[str(cls)] = pred
                sorted_tmp = []
                for i in range(1, 35):
                    sorted_tmp.append(tmp[str(i)])
                self.image_prediction.append(sorted_tmp)
            except Exception as e:
                self.image_prediction.append([0]*34)

        _, scores = self.text_classifier.eval()
        scores = [softmax(s) for s in scores]
        text_only = [np.argmax(s) for s in scores]

        # 최종 prediction 값
        text = np.array(scores)
        img = np.array(self.image_prediction)

        img_only = [np.argmax(i) for i in img]
        prediction_ = (text * 0.25) + (img * 0.75)
        final_prediction = [np.argmax(pred) for pred in prediction_]

        self.label = np.array(self.label)
        text_acc = float(sum(text_only == self.label)) / len(self.label)
        img_acc = float(sum(img_only == self.label)) / len(self.label)
        print('텍스트만 사용했을 때 Accuracy : ', text_acc)
        print('이미지만 사용했을 때 Accuracy : ', img_acc)
        print('텍스트와 이미지를 사용했을 때 Accuracy : ', np.mean(final_prediction == self.label))

    def predict(self, product_name, image_url):
        category_db = CategoryManager()
        text_score = self.text_classifier.predict(product_name=product_name)
        try:
            request.urlretrieve(image_url, 'tmp.jpg')
            image_score = self.image_classifier.predict('tmp.jpg')
        except:
            tmp = DataFrame({'score': [0] * len(self.image_classifier.labels), 'classes': self.image_classifier.labels})
            image_score = [pre[1] for pre in tmp.values]

        text = np.array(text_score)
        img = np.array(image_score)

        prediction_ = (text * 0.25) + (img * 0.75)
        cate_id = np.argmax(prediction_)
        res = category_db.retrieve_cate_name_by_cate_id(int(cate_id)+1)
        print('FINAL CATEGORY : {}'.format(res))
        img = Image.open('tmp.jpg')
        img.show()

        if os.path.exists('tmp.jpg'):
            os.remove('tmp.jpg')

if __name__ == '__main__':
    cls = Classifier()
    start = time.time()
    cls.evaluate()
    end = time.time()
    print('Total time : ', end - start)  # 0.827

    while True:
        product_name = input('Product Name(exit to q or Q) : ')
        image_url = input('Image URL : ')
        if product_name == 'Q' or product_name == 'q':
            break
        cls.predict(product_name=product_name, image_url=image_url)

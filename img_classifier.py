import tensorflow as tf
import os
import pickle

from pandas import DataFrame

save_path = 'image_classifier/predictions/'


class ImageClassifier:
    def __init__(self):
        with tf.gfile.FastGFile('image_classifier/graph/output_graph.pb', 'rb') as fp:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fp.read())
            tf.import_graph_def(graph_def, name='')
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.image_prediction = []
        self.logits = self.sess.graph.get_tensor_by_name('final_result:0')
        self.labels = [int(line.rstrip()) for line in tf.gfile.GFile('image_classifier/graph/output_labels.txt')]

    def make_evaluation_data(self):
        with open('../data_/test.txt', 'r', encoding='utf-8') as f:
            test_data = f.readlines()

        for data in test_data:
            try:
                product_seq = data.split(',')[2].strip()
                category_seq = data.split(',')[1].strip()
                file_name = str(product_seq) + '.jpg'
                img_path = 'img/test/' + str(category_seq) + '/' + file_name
                image = tf.gfile.FastGFile(img_path, 'rb').read()
                prediction = self.sess.run(self.logits, {'DecodeJpeg/contents:0': image})[0]
                tmp = DataFrame({'score': prediction, 'classes': self.labels})
                tmp = tmp.sort_values(by='classes')
                self.image_prediction.append([pre[1] for pre in tmp.values])
            except Exception:
                tmp = DataFrame({'score': [0] * len(self.labels), 'classes': self.labels})
                self.image_prediction.append([pre[1] for pre in tmp.values])

        with open(os.path.join(save_path, 'image_prediction'), 'wb') as f:
            pickle.dump(self.image_prediction, f)

    def evaluate(self):
        if os.path.exists(os.path.join(save_path, 'image_prediction')):
            with open(os.path.join(save_path, 'image_prediction'), 'rb') as f:
                self.image_prediction = pickle.load(f)

    def predict(self, img):
        image = tf.gfile.FastGFile(img, 'rb').read()
        prediction = self.sess.run(self.logits, {'DecodeJpeg/contents:0': image})[0]
        tmp = DataFrame({'score': prediction, 'classes': self.labels})
        tmp = tmp.sort_values(by='classes')
        return [pre[1] for pre in tmp.values]

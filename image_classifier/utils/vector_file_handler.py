import os

import numpy as np
import tensorflow as tf
import pickle
from . import configs
from .ops import save_pkl_file


# classes
class VectorSaver:
    def __init__(self, vector_dir_path):
        path = os.path.join(vector_dir_path, configs.VECTORS_FILE_NAME)
        self.writer = tf.python_io.TFRecordWriter(path)

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def add_vector(self, name, vector, flatten=True):
        if flatten and len(vector.shape) != 1:
            vector = vector.flatten()

        example = tf.train.Example(features=tf.train.Features(feature={
                                'name': self._bytes_feature(name.encode()),
                                'vector_raw': self._bytes_feature(vector.tostring())}))

        self.writer.write(example.SerializeToString())


class VectorLoader:
    def __init__(self, vector_dir_path):
        path = os.path.join(vector_dir_path, configs.VECTORS_FILE_NAME)
        self.record_iterator = tf.python_io.tf_record_iterator(path=path)

    def get_vectors_generator(self):
        for string_record in self.record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            
            name = (example.features.feature['name']
                                          .bytes_list
                                          .value[0])
            
            vector_raw = (example.features.feature['vector_raw']
                                        .bytes_list
                                        .value[0])
            
            name = name.decode()
            vector = np.fromstring(vector_raw, dtype=np.float32)
            
            yield name, vector


# function
def get_vector_dir_path(vectors_path, model_type):
    if model_type == 'auto':
        path = os.path.join(vectors_path, configs.ATUO_VECTORS_FOLDER_NAME)
    elif model_type == 'iv4':
        path = os.path.join(vectors_path, configs.I4_VECTORS_FOLDER_NAME)
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def create_metadata_file(vector_dir_path, args):
    save_pkl_file(args, os.path.join(vector_dir_path,
                                     configs.METADATA_FILE_NAME))


def mark_type(vector_dir_path, is_pretrained):
    with open(os.path.join(vector_dir_path, configs.TYPE_FILE_NAME),
              'w') as txt_file:
        if is_pretrained:
            txt_file.write('pretrained')
        else:
            txt_file.write('autoencoder')


def establish_vectors_folder(vectors_path, is_pretrained, model_type='iv4'):
    path = get_vector_dir_path(vectors_path, model_type)
    mark_type(path, is_pretrained)

    return path


def save_vec2list(vector_actual_path=None):
    vector_generator = VectorLoader(vector_actual_path).get_vectors_generator()

    vector_list = []
    for name, vec in vector_generator:
        vector_list.append((name, vec))

    with open(vector_actual_path + '/vec2list.pickle', 'wb') as f:
        pickle.dump(vector_list, f)
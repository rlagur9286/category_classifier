import logging
from six.moves import cPickle
import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean


# functions
def get_pkl_file(file_path):
    with open(file_path, 'rb') as pkl_file:
        return cPickle.load(pkl_file)


def save_pkl_file(data, file_path):
    with open(file_path, 'wb') as pkl_file:
        cPickle.dump(data, pkl_file)


def load_image(image_path, size=None, failure_image=None, allow_non_rgb=False):
    try:
        image = Image.open(image_path)
    except Exception as e:
        logging.warning('error loading image at "{}" with ' \
                        'exception "{}"'.format(image_path, e))
        return failure_image

    if size:
        image = image.resize(size)

    image = np.array(image)

    if not allow_non_rgb:
        if len(image.shape) != 3 or \
           (len(image.shape) == 3 and image.shape[-1] != 3):
            logging.warning('image at "{}" isn\'t RGB, therefore ' \
                            'not using it'.format(image_path))
            return failure_image

    return image


def get_file_list(dir_path=None, output=None, prefix=None):
    with open(output, 'w') as f:
        for root, dirs, files in os.walk(dir_path):
                for file in files:
                    f.write(file + '\n')


def walk(top, depth=None, followlinks=False):
    top = top.rstrip(os.path.sep)
    assert os.path.isdir(top)
    num_sep = top.count(os.path.sep)
    for root, dirs, files in os.walk(top, followlinks=followlinks):
        yield root, dirs, files
        if depth is not None and num_sep + depth <= root.count(os.path.sep):
            del dirs[:]
        pass
    pass


def get_similarity_func(name='cos'):
    name = name.lower()
    if name in ['cosine', 'cos']:
        return cosine
    elif name in ['euclidean', 'euc']:
        return euclidean
    else:
        raise 'Unknown distance function: {}'.format(name)

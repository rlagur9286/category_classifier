from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from image_classifier.utils import configs
from image_classifier.utils.vector_file_handler import VectorSaver
from image_classifier.utils.vector_file_handler import establish_vectors_folder

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


class Incept_v4_Trainer(object):
    def __init__(self, image_dir='images/', output_graph='image_classifier/graph/output_graph.pb',
                 output_labels='image_classifier/graph/output_labels.txt',
                 summaries_dir='image_classifier/tmp/retrain_logs', how_many_training_steps=1000, learning_rate=1e-3,
                 testing_percentage=5, eval_step_interval=100, train_batch_size=64, vector_path='image_classifier/vectors',
                 test_batch_size=32, validation_batch_size=32, print_misclassified_test_images=False,
                 model_dir='image_classifier/models', bottleneck_dir='image_classifier/bottleneck',
                 final_tensor_name='final_result', flip_left_right=False, random_crop=False, random_scale=0,
                 random_brightness=0, check_point_path='image_classifier/check_point', max_ckpts_to_keep=3, validation_percentage=5):

        self.image_dir = image_dir
        self.output_graph = output_graph
        self.output_labels = output_labels
        self.summaries_dir = summaries_dir
        self.how_many_training_steps = how_many_training_steps
        self.learning_rate = learning_rate
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage
        self.eval_step_interval = eval_step_interval
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.validation_batch_size = validation_batch_size
        self.print_misclassified_test_images = print_misclassified_test_images
        self.model_dir = model_dir
        self.bottleneck_dir = bottleneck_dir
        self.final_tensor_name = final_tensor_name
        self.flip_left_right = flip_left_right
        self.random_crop = random_crop
        self.random_scale = random_scale
        self.random_brightness = random_brightness
        self.check_point_path = check_point_path
        self.max_ckpts_to_keep = max_ckpts_to_keep
        self.vector_path = vector_path

    def create_image_lists(self, image_dir, testing_percentage, validation_percentage):

        if not gfile.Exists(image_dir):
            print("Image directory '" + image_dir + "' not found.")
            return None
        result = {}

        sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
        # The root directory comes first, so skip it.
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == image_dir:
                continue
            print("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
                file_list.extend(gfile.Glob(file_glob))
            if not file_list:
                print('No files found')
                continue
            if len(file_list) < 20:
                print('WARNING: Folder has less than 20 images, which may cause issues.')
            elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
                print('WARNING: Folder {} has more than {} images. Some images will '
                      'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            training_images = []
            testing_images = []
            validation_images = []
            for file_name in file_list:
                base_name = os.path.basename(file_name)
                # We want to ignore anything after '_nohash_' in the file name when
                # deciding which set to put an image in, the data set creator has a way of
                # grouping photos that are close variations of each other. For example
                # this is used in the plant disease data set to group multiple pictures of
                # the same leaf.
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                # This looks a bit magical, but we need to decide whether this file should
                # go into the training, testing, or validation sets, and we want to keep
                # existing files in the same set even if more files are subsequently
                # added.
                # To do that, we need a stable way of deciding based on just the file name
                # itself, so we do a hash of that and then use that to generate a
                # probability value that we use to assign it.
                hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
                percentage_hash = ((int(hash_name_hashed, 16) %
                                    (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                                   (100.0 / MAX_NUM_IMAGES_PER_CLASS))
                if percentage_hash < validation_percentage:
                    validation_images.append(base_name)
                elif percentage_hash < (testing_percentage + validation_percentage):
                    testing_images.append(base_name)
                else:
                    training_images.append(base_name)
            result[label_name] = {
                'dir': dir_name,
                'training': training_images,
                'testing': testing_images,
                'validation': validation_images,
            }
        return result

    def get_image_path(self, image_lists, label_name, index, image_dir, category):
        if label_name not in image_lists:
            tf.logging.fatal('Label does not exist %s.', label_name)
        label_lists = image_lists[label_name]
        if category not in label_lists:
            tf.logging.fatal('Category does not exist %s.', category)
        category_list = label_lists[category]
        if not category_list:
            tf.logging.fatal('Label %s has no images in the category %s.',
                             label_name, category)

        try:
            mod_index = index % len(category_list)
        except Exception as e:
            print(image_lists)
        base_name = category_list[mod_index]
        sub_dir = label_lists['dir']
        if '.jpg' not in os.path.join(image_dir, sub_dir, base_name).split('/')[0] and not os.path.exists(os.path.join(image_dir, sub_dir, base_name).split('/')[0]):
            os.makedirs(os.path.join(image_dir, sub_dir, base_name).split('/')[0])
        full_path = os.path.join(image_dir, sub_dir, base_name)
        return full_path

    def get_bottleneck_path(self, image_lists, label_name, index, bottleneck_dir, category):
        return self.get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '.txt'

    def create_inception_graph(self):
        with tf.Session() as sess:
            model_filename = os.path.join(
                self.model_dir, 'classify_image_graph_def.pb')
            with gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                    tf.import_graph_def(graph_def, name='', return_elements=[
                        BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                        RESIZED_INPUT_TENSOR_NAME]))
        return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

    def run_bottleneck_on_image(self, sess, image_data, image_data_tensor, bottleneck_tensor):
        bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    def ensure_dir_exists(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def write_list_of_floats_to_file(self, list_of_floats , file_path):
        s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *list_of_floats)
        with open(file_path, 'wb') as f:
            f.write(s)

    def read_list_of_floats_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
            return list(s)

    def get_or_create_bottleneck(self, sess, image_lists, label_name, index, image_dir,
                                 category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
        label_lists = image_lists[label_name]
        sub_dir = label_lists['dir']
        sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
        self.ensure_dir_exists(sub_dir_path)
        bottleneck_path = self.get_bottleneck_path(image_lists, label_name, index,
                                                   bottleneck_dir, category)
        if not os.path.exists(bottleneck_path):
            # print('Creating bottleneck at ' + bottleneck_path)
            image_path = self.get_image_path(image_lists, label_name, index, image_dir,
                                             category)
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            try:
                image_data = gfile.FastGFile(image_path, 'rb').read()
                bottleneck_values = self.run_bottleneck_on_image(sess, image_data,
                                                                 jpeg_data_tensor,
                                                                 bottleneck_tensor)
            except:
                try:
                    print('IN')
                    im = Image.open(image_path)
                    im.convert('RGB').save(image_path, 'JPEG')
                    image_data = gfile.FastGFile(image_path, 'rb').read()
                    bottleneck_values = self.run_bottleneck_on_image(sess, image_data,
                                                                     jpeg_data_tensor,
                                                                     bottleneck_tensor)
                    print('OUT')
                except:
                    print('GIF', image_path)
                    return
            bottleneck_string = ','.join(str(x) for x in bottleneck_values)
            with open(bottleneck_path, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        return bottleneck_values


    def cache_bottlenecks(self, sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
        how_many_bottlenecks = 0
        self.ensure_dir_exists(bottleneck_dir)
        for label_name, label_lists in image_lists.items():
            for category in ['training', 'testing', 'validation']:
                category_list = label_lists[category]
                for index, unused_base_name in enumerate(category_list):
                    self.get_or_create_bottleneck(sess, image_lists, label_name, index,
                                                  image_dir, category, bottleneck_dir,
                                                  jpeg_data_tensor, bottleneck_tensor)
                    how_many_bottlenecks += 1
                    if how_many_bottlenecks % 100 == 0:
                        print(str(how_many_bottlenecks) + ' bottleneck files created.')


    def get_random_cached_bottlenecks(self, sess, image_lists, how_many, category,
                                      bottleneck_dir, image_dir, jpeg_data_tensor,
                                      bottleneck_tensor):
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        filenames = []
        if how_many >= 0:
            # Retrieve a random sample of bottlenecks.
            for unused_i in range(how_many):
                label_index = random.randrange(class_count)
                label_name = list(image_lists.keys())[label_index]
                image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
                image_name = self.get_image_path(image_lists, label_name, image_index,
                                                 image_dir, category)
                bottleneck = self.get_or_create_bottleneck(sess, image_lists, label_name,
                                                           image_index, image_dir, category,
                                                           bottleneck_dir, jpeg_data_tensor,
                                                           bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
        else:
            for label_index, label_name in enumerate(image_lists.keys()):
                for image_index, image_name in enumerate(
                        image_lists[label_name][category]):
                    image_name = self.get_image_path(image_lists, label_name, image_index,
                                                     image_dir, category)
                    try:
                        bottleneck = self.get_or_create_bottleneck(sess, image_lists, label_name,
                                                                   image_index, image_dir, category,
                                                                   bottleneck_dir, jpeg_data_tensor,
                                                                   bottleneck_tensor)
                    except Exception as e:
                        continue
                    ground_truth = np.zeros(class_count, dtype=np.float32)
                    ground_truth[label_index] = 1.0
                    bottlenecks.append(bottleneck)
                    ground_truths.append(ground_truth)
                    filenames.append(image_name)
        return bottlenecks, ground_truths, filenames

    def get_random_distorted_bottlenecks(self, sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
                                         distorted_image, resized_input_tensor, bottleneck_tensor):
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_path = self.get_image_path(image_lists, label_name, image_index, image_dir,
                                             category)
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            jpeg_data = gfile.FastGFile(image_path, 'rb').read()
            # Note that we materialize the distorted_image_data as a numpy array before
            # sending running inference on the image. This involves 2 memory copies and
            # might be optimized in other implementations.
            distorted_image_data = sess.run(distorted_image,
                                            {input_jpeg_tensor: jpeg_data})
            bottleneck = self.run_bottleneck_on_image(sess, distorted_image_data,
                                                      resized_input_tensor,
                                                      bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
        return bottlenecks, ground_truths

    def should_distort_images(self, flip_left_right, random_crop, random_scale, random_brightness):
        return flip_left_right or (random_crop != 0) or (random_scale != 0) or (random_brightness != 0)

    def add_input_distortions(self, flip_left_right, random_crop, random_scale, random_brightness):
        jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        margin_scale = 1.0 + (random_crop / 100.0)
        resize_scale = 1.0 + (random_scale / 100.0)
        margin_scale_value = tf.constant(margin_scale)
        resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                               minval=1.0,
                                               maxval=resize_scale)
        scale_value = tf.multiply(margin_scale_value, resize_scale_value)
        precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
        precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
        precrop_shape = tf.stack([precrop_height, precrop_width])
        precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
        precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                    precrop_shape_as_int)
        precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
        cropped_image = tf.random_crop(precropped_image_3d,
                                       [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                        MODEL_INPUT_DEPTH])
        if flip_left_right:
            flipped_image = tf.image.random_flip_left_right(cropped_image)
        else:
            flipped_image = cropped_image
        brightness_min = 1.0 - (random_brightness / 100.0)
        brightness_max = 1.0 + (random_brightness / 100.0)
        brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                             minval=brightness_min,
                                             maxval=brightness_max)
        brightened_image = tf.multiply(flipped_image, brightness_value)
        distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
        return jpeg_data, distort_result

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def add_final_training_ops(self, class_count, final_tensor_name, bottleneck_tensor):
        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
                name='BottleneckInputPlaceholder')
            ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')

        layer_name = 'final_training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')
                self.variable_summaries(layer_weights)
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
                self.variable_summaries(layer_biases)
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activations', logits)

        final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
        tf.summary.histogram('activations', final_tensor)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
                cross_entropy_mean)

        return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor

    def add_evaluation_step(self, result_tensor, ground_truth_tensor):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                prediction = tf.argmax(result_tensor, 1)
                correct_prediction = tf.equal(
                    prediction, tf.argmax(ground_truth_tensor, 1))
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        return evaluation_step, prediction

    def do_train(self):
        # Setup the directory we'll write summaries to for TensorBoard
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)

        # Set up the pre-trained graph.
        graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = self.create_inception_graph()

        # Look at the folder structure, and create lists of all the images.
        image_lists = self.create_image_lists(self.image_dir, self.testing_percentage, self.validation_percentage)
        if image_lists is False:
            return False
        class_count = len(image_lists.keys())
        if class_count == 0:
            print('No valid folders of images found at ' + self.image_dir)
            return -1
        if class_count == 1:
            print('Only one valid folder of images found at ' + self.image_dir +
                  ' - multiple classes are needed for classification.')
            return -1

        # See if the command-line flags mean we're applying any distortions.
        do_distort_images = self.should_distort_images(
            self.flip_left_right, self.random_crop, self.random_scale,
            self.random_brightness)
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        if do_distort_images:
            # We will be applying distortions, so setup the operations we'll need.
            distorted_jpeg_data_tensor, distorted_image_tensor = self.add_input_distortions(
                self.flip_left_right, self.random_crop, self.random_scale,
                self.random_brightness)
        else:
            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            self.cache_bottlenecks(sess, image_lists, self.image_dir, self.bottleneck_dir,
                                   jpeg_data_tensor, bottleneck_tensor)

        # Add the new layer that we'll be training.
        (train_step, cross_entropy, bottleneck_input, ground_truth_input,
         final_tensor) = self.add_final_training_ops(len(image_lists.keys()),
                                                     self.final_tensor_name,
                                                     bottleneck_tensor)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, prediction = self.add_evaluation_step(
            final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
        merged = tf.summary.merge_all()

        # Set up all our weights to their initial default values.
        saver = tf.train.Saver(max_to_keep=self.max_ckpts_to_keep)

        init = tf.global_variables_initializer()
        sess.run(init)

        # Run the training for as many cycles as requested on the command line.
        for i in range(self.how_many_training_steps):
            # Get a batch of input bottleneck values, either calculated fresh every time
            # with distortions applied, or from the cache stored on disk.
            if do_distort_images:
                train_bottlenecks, train_ground_truth = self.get_random_distorted_bottlenecks(
                    sess, image_lists, self.train_batch_size, 'training',
                    self.image_dir, distorted_jpeg_data_tensor,
                    distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                train_bottlenecks, train_ground_truth, _ = self.get_random_cached_bottlenecks(
                    sess, image_lists, self.train_batch_size, 'training',
                    self.bottleneck_dir, self.image_dir, jpeg_data_tensor,
                    bottleneck_tensor)
            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run([merged, train_step],
                                        feed_dict={bottleneck_input: train_bottlenecks,
                                                   ground_truth_input: train_ground_truth})

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == self.how_many_training_steps)
            if (i % self.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                                train_accuracy * 100))
                print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                           cross_entropy_value))
                validation_bottlenecks, validation_ground_truth, _ = (
                    self.get_random_cached_bottlenecks(
                        sess, image_lists, self.validation_batch_size, 'validation',
                        self.bottleneck_dir, self.image_dir, jpeg_data_tensor,
                        bottleneck_tensor))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (datetime.now(), i, validation_accuracy * 100,
                       len(validation_bottlenecks)))

                _ = saver.save(sess, os.path.join(self.check_point_path, configs.MODEL_CHECKPOINT_NAME.format(i)))

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        test_bottlenecks, test_ground_truth, test_filenames = (
            self.get_random_cached_bottlenecks(sess, image_lists, self.test_batch_size,
                                               'testing', self.bottleneck_dir,
                                               self.image_dir, jpeg_data_tensor,
                                               bottleneck_tensor))
        test_accuracy, predictions = sess.run(
            [evaluation_step, prediction],
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%% (N=%d)' % (
            test_accuracy * 100, len(test_bottlenecks)))

        if self.print_misclassified_test_images:
            print('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    print('%70s  %s' % (test_filename, image_lists.keys()[predictions[i]]))

        # Write out the trained graph and labels with the weights stored as constants.
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [self.final_tensor_name])
        with gfile.FastGFile(self.output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        with gfile.FastGFile(self.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

    def do_train_with_GPU(self, gpu_list=None):
        # Setup the directory we'll write summaries to for TensorBoard
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)

        for gpu in gpu_list:
            with tf.device(gpu):
                # Set up the pre-trained graph.
                graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = self.create_inception_graph()

                # Look at the folder structure, and create lists of all the images.
                image_lists = self.create_image_lists(self.image_dir, self.testing_percentage, self.validation_percentage)
                if image_lists is False:
                    return False
                class_count = len(image_lists.keys())
                if class_count == 0:
                    print('No valid folders of images found at ' + self.image_dir)
                    return -1
                if class_count == 1:
                    print('Only one valid folder of images found at ' + self.image_dir +
                          ' - multiple classes are needed for classification.')
                    return -1

                # See if the command-line flags mean we're applying any distortions.
                do_distort_images = self.should_distort_images(
                    self.flip_left_right, self.random_crop, self.random_scale,
                    self.random_brightness)
                config = tf.ConfigProto(allow_soft_placement=True)
                sess = tf.Session(config=config)

                if do_distort_images:
                    # We will be applying distortions, so setup the operations we'll need.
                    distorted_jpeg_data_tensor, distorted_image_tensor = self.add_input_distortions(
                        self.flip_left_right, self.random_crop, self.random_scale,
                        self.random_brightness)
                else:
                    # We'll make sure we've calculated the 'bottleneck' image summaries and
                    # cached them on disk.
                    self.cache_bottlenecks(sess, image_lists, self.image_dir, self.bottleneck_dir,
                                           jpeg_data_tensor, bottleneck_tensor)

                # Add the new layer that we'll be training.
                (train_step, cross_entropy, bottleneck_input, ground_truth_input,
                 final_tensor) = self.add_final_training_ops(len(image_lists.keys()),
                                                             self.final_tensor_name,
                                                             bottleneck_tensor)

                # Create the operations we need to evaluate the accuracy of our new layer.
                evaluation_step, prediction = self.add_evaluation_step(
                    final_tensor, ground_truth_input)

                # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
                merged = tf.summary.merge_all()

                # Set up all our weights to their initial default values.
                saver = tf.train.Saver(max_to_keep=self.max_ckpts_to_keep)

                init = tf.global_variables_initializer()
                sess.run(init)

                # Run the training for as many cycles as requested on the command line.
                for i in range(self.how_many_training_steps):
                    # Get a batch of input bottleneck values, either calculated fresh every time
                    # with distortions applied, or from the cache stored on disk.
                    if do_distort_images:
                        train_bottlenecks, train_ground_truth = self.get_random_distorted_bottlenecks(
                            sess, image_lists, self.train_batch_size, 'training',
                            self.image_dir, distorted_jpeg_data_tensor,
                            distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
                    else:
                        train_bottlenecks, train_ground_truth, _ = self.get_random_cached_bottlenecks(
                            sess, image_lists, self.train_batch_size, 'training',
                            self.bottleneck_dir, self.image_dir, jpeg_data_tensor,
                            bottleneck_tensor)
                    # Feed the bottlenecks and ground truth into the graph, and run a training
                    # step. Capture training summaries for TensorBoard with the `merged` op.
                    train_summary, _ = sess.run([merged, train_step],
                                                feed_dict={bottleneck_input: train_bottlenecks,
                                                           ground_truth_input: train_ground_truth})

                    # Every so often, print out how well the graph is training.
                    is_last_step = (i + 1 == self.how_many_training_steps)
                    if (i % self.eval_step_interval) == 0 or is_last_step:
                        train_accuracy, cross_entropy_value = sess.run(
                            [evaluation_step, cross_entropy],
                            feed_dict={bottleneck_input: train_bottlenecks,
                                       ground_truth_input: train_ground_truth})
                        print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                                        train_accuracy * 100))
                        print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                                   cross_entropy_value))
                        validation_bottlenecks, validation_ground_truth, _ = (
                            self.get_random_cached_bottlenecks(
                                sess, image_lists, self.validation_batch_size, 'validation',
                                self.bottleneck_dir, self.image_dir, jpeg_data_tensor,
                                bottleneck_tensor))
                        # Run a validation step and capture training summaries for TensorBoard
                        # with the `merged` op.
                        validation_summary, validation_accuracy = sess.run(
                            [merged, evaluation_step],
                            feed_dict={bottleneck_input: validation_bottlenecks,
                                       ground_truth_input: validation_ground_truth})
                        print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                              (datetime.now(), i, validation_accuracy * 100,
                               len(validation_bottlenecks)))

                        _ = saver.save(sess, os.path.join(self.check_point_path, configs.MODEL_CHECKPOINT_NAME.format(i)))

                # We've completed all our training, so run a final test evaluation on
                # some new images we haven't used before.
                test_bottlenecks, test_ground_truth, test_filenames = (
                    self.get_random_cached_bottlenecks(sess, image_lists, self.test_batch_size,
                                                       'testing', self.bottleneck_dir,
                                                       self.image_dir, jpeg_data_tensor,
                                                       bottleneck_tensor))
                test_accuracy, predictions = sess.run(
                    [evaluation_step, prediction],
                    feed_dict={bottleneck_input: test_bottlenecks,
                               ground_truth_input: test_ground_truth})
                print('Final test accuracy = %.1f%% (N=%d)' % (
                    test_accuracy * 100, len(test_bottlenecks)))

                if self.print_misclassified_test_images:
                    print('=== MISCLASSIFIED TEST IMAGES ===')
                    for i, test_filename in enumerate(test_filenames):
                        if predictions[i] != test_ground_truth[i].argmax():
                            print('%70s  %s' % (test_filename, image_lists.keys()[predictions[i]]))

                # Write out the trained graph and labels with the weights stored as constants.
                output_graph_def = graph_util.convert_variables_to_constants(
                    sess, graph.as_graph_def(), [self.final_tensor_name])
                with gfile.FastGFile(self.output_graph, 'wb') as f:
                    f.write(output_graph_def.SerializeToString())
                with gfile.FastGFile(self.output_labels, 'w') as f:
                    f.write('\n'.join(image_lists.keys()) + '\n')

    def vectorize(self):
        image_paths = []
        for (paths, dir, files) in os.walk(self.image_dir):
            for file in files:
                if (file != ".DS_Store"):
                    image_paths.append(paths + "/" + file)

        with tf.gfile.FastGFile(self.output_graph, 'rb') as fp:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fp.read())
            tf.import_graph_def(graph_def, name='')
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            bottleneck = sess.graph.get_tensor_by_name('input/BottleneckInputPlaceholder:0')
            vectors_path = establish_vectors_folder(self.vector_path, False)
            vector_saver = VectorSaver(vectors_path)

            for img in image_paths:
                try:
                    image = tf.gfile.FastGFile(img, 'rb').read()
                    vector = sess.run(bottleneck, {'DecodeJpeg/contents:0': image})
                    vector_saver.add_vector(img, vector[0])
                except:
                    print(img)
                    continue
        return vectors_path

    def vectorize_with_GPU(self, gpu_list=None):
        image_paths = []
        for (paths, dir, files) in os.walk(self.image_dir):
            for file in files:
                if (file != ".DS_Store"):
                    image_paths.append(paths + "/" + file)

        with tf.gfile.FastGFile(self.output_graph, 'rb') as fp:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fp.read())
            tf.import_graph_def(graph_def, name='')
        config = tf.ConfigProto(allow_soft_placement=True)
        for gpu in gpu_list:
            with tf.device(gpu):
                with tf.Session(config=config) as sess:
                    bottleneck = sess.graph.get_tensor_by_name('input/BottleneckInputPlaceholder:0')
                    vectors_path = establish_vectors_folder(self.vector_path, False)
                    vector_saver = VectorSaver(vectors_path)

                    for img in image_paths:
                        try:
                            image = tf.gfile.FastGFile(img, 'rb').read()
                            vector = sess.run(bottleneck, {'DecodeJpeg/contents:0': image})
                            vector_saver.add_vector(img, vector[0])
                        except:
                            print(img)
                            continue
        return vectors_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='python Incept_v4_Trainer.py')
    parser.add_argument('--image_dir', default='img/train/')
    parser.add_argument('--output_graph', default='image_classifier/graph/output_graph.pb')
    parser.add_argument('--output_labels', default='image_classifier/graph/output_labels.txt')
    parser.add_argument('--summaries_dir', default='image_classifier/tmp/retrain_logs')
    parser.add_argument('--how_many_training_steps', default=50000)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--testing_percentage', default=10)
    parser.add_argument('--validation_percentage', default=10)
    parser.add_argument('--eval_step_interval', default=100)
    parser.add_argument('--train_batch_size', default=128)
    parser.add_argument('--test_batch_size', default=-1, help='-1 means All images')
    parser.add_argument('--validation_batch_size', default=128)
    parser.add_argument('--print_misclassified_test_images', default=False)
    parser.add_argument('--model_dir', default='image_classifier/models')
    parser.add_argument('--bottleneck_dir', default='image_classifier/bottleneck')
    parser.add_argument('--final_tensor_name', default='final_result')
    # Option for making distorted Images
    parser.add_argument('--flip_left_right', default=False)
    parser.add_argument('--random_crop', default=0)
    parser.add_argument('--random_scale', default=0)
    parser.add_argument('--random_brightness', default=0)

    parser.add_argument('--check_point_path', default='image_classifier/check_point')
    parser.add_argument('--max_ckpts_to_keep', default=5)
    ARGS = parser.parse_args()
    t4 = Incept_v4_Trainer(image_dir=ARGS.image_dir, output_graph=ARGS.output_graph, output_labels=ARGS.output_labels,
                           summaries_dir=ARGS.summaries_dir, how_many_training_steps=ARGS.how_many_training_steps,
                           learning_rate=ARGS.learning_rate, testing_percentage=ARGS.testing_percentage,
                           eval_step_interval=ARGS.eval_step_interval,
                           train_batch_size=ARGS.train_batch_size, test_batch_size=ARGS.test_batch_size,
                           validation_batch_size=ARGS.validation_batch_size,
                           print_misclassified_test_images=ARGS.print_misclassified_test_images, model_dir=ARGS.model_dir,
                           bottleneck_dir=ARGS.bottleneck_dir, final_tensor_name=ARGS.final_tensor_name,
                           flip_left_right=ARGS.flip_left_right, random_crop=ARGS.random_crop,
                           random_scale=ARGS.random_scale, random_brightness=ARGS.random_brightness,
                           check_point_path=ARGS.check_point_path, max_ckpts_to_keep=ARGS.max_ckpts_to_keep,
                           validation_percentage=ARGS.validation_percentage)
    t4.do_train_with_GPU(gpu_list=['/gpu:0'])

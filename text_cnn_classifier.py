import tensorflow as tf
import numpy as np
import os
import datetime
import time

from konlpy.tag import Twitter
from tensorflow.contrib import learn
from multi_class_data_loader import MultiClassDataLoader
from utils.database import ProductManager
from utils.database import CategoryManager


class WordDataProcessor(object):

    def vocab_processor(_, *texts):
        max_document_length = 0
        for text in texts:
            max_doc_len = max([len(line.split(" ")) for line in text])
            if max_doc_len > max_document_length:
                max_document_length = max_doc_len

        return learn.preprocessing.VocabularyProcessor(max_document_length)

    def restore_vocab_processor(_, vocab_path):
        return learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    def clean_data(_, string):
        """
        형태소(DHA) 분석된 결과로 학습할 것이므로 데이타 정제는 필요 없음
        """
        if ":" not in string:
            string = string.strip().lower()
        return string

pos_tagger = Twitter()
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

data_loader = MultiClassDataLoader(tf.flags, WordDataProcessor())
data_loader.define_flags()

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    주어진 batch size 만큼씩 데이터 yield
    :param data: 전체 데이터
    :param batch_size: batch size
    :param num_epochs: 전체 epoch
    :param shuffle: 섞을지 말지
    :return: batch size 만큼의 dataset
    """

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def chunks(x, n):
    return [x[i:i + n] for i in range(0, len(x), n)]


def splits(doc):
    doc = doc.replace('[', '').replace(']', ' ').replace('(', '').replace(')', ' ').replace('\'', ' ').replace('\"',
                                                                                                               ' ')
    doc = doc.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').replace('/', ' ') \
        .replace('-', ' ').replace('_', ' ').replace('+', ' ').replace('~', ' ')
    return doc


def tokenize(doc):
    result = []
    for t in pos_tagger.pos(doc, norm=True):
        if t[1] not in ['Punctuation']:
            result.append('/'.join(t))
    return ' '.join(result)


class TextCNNClassifier:
    def __init__(self):
        self.accuracy = 0
        # 특별하게 checkpoint dir이 주어지지 않았다면 가장 최근 checkpoint Load
        if FLAGS.checkpoint_dir == "":
            all_subdirs = ["./runs/" + d for d in os.listdir('./runs/.') if os.path.isdir("./runs/" + d)]
            latest_subdir = max(all_subdirs, key=os.path.getmtime)
            FLAGS.checkpoint_dir = latest_subdir + "/checkpoint"

        # 단어 사전 불러오기
        vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
        self.vocab_processor = data_loader.restore_vocab_processor(vocab_path)

        # 그래프 그리고 checkpoint 데이터 불러오기
        self.graph = tf.Graph()
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            self.sess.run(tf.global_variables_initializer())

            checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)

    def read_data(self):
        """
        DB로부터 데이터를 읽어 온 후 앞 10%는 Test set으로 enl 90%는 Train set으로 나누어서 저장
        :return: 
        """

        product_db = ProductManager()
        category_db = CategoryManager()
        # DB로부터 모든 데이터 불러오기
        products = product_db.retrieve_products_by_cate()

        # 데이터 중 뒤 90% 학습 데이터 셋으로 따로 저장
        train_data = products[len(products)//10:]
        train_data = [(data.get('product_name'), data.get('category_seq'), data.get('product_seq')) for data in train_data]
        train_data = [(tokenize(splits(row[0])), row[1], row[2]) for row in train_data]

        # 데이터 중 앞 10% 테스트 셋으로 따로 저장
        test_data = products[:len(products)//10]
        test_data = [(data.get('product_name'), data.get('category_seq'), data.get('product_seq')) for data in test_data]
        test_data = [(tokenize(splits(row[0])), row[1], row[2]) for row in test_data]

        # 나중에 쓰기 쉽게 .txt 형식으로 저장
        with open('data_/train.txt', 'w', encoding='utf-8') as f:
            for data in train_data:
                f.write(''.join(data[0]) + ',' + str(data[1]) + ',' + str(data[2]) + '\n')

        with open('data_/test.txt', 'w', encoding='utf-8') as f:
            for data in test_data:
                f.write(''.join(data[0]) + ',' + str(data[1]) + ',' + str(data[2]) + '\n')

        with open('data_/cls.txt', 'w', encoding='utf-8') as f:
            for cls in category_db.retrieve_cate_list():
                f.write(str(cls.get('category_seq')) + '\n')

    def get_checkpoint(self, out_dir):
        """
        checkpoint 디렉토리 없다면 생성
        :param out_dir: 
        :return: 
        """
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoint'))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        return checkpoint_dir

    def train(self):
        print("\nParameters:")
        for attr, value in sorted(FLAGS.__flags.items()):
            print("{}={}".format(attr.upper(), value))
        print("")

        # Load data
        print("Loading data...")
        x_train, y_train, x_dev, y_dev = data_loader.prepare_data()
        vocab_processor = data_loader.vocab_processor

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)

            sess = tf.Session(config=session_conf)
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            with sess.as_default():
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(0.005)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = self.get_checkpoint(out_dir)
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables())

                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, loss, accuracy = sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy = sess.run(
                        [global_step, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    return accuracy

                # Generate batches
                batches = batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                with tf.device('/gpu:0'):
                    for batch in batches:
                        x_batch, y_batch = zip(*batch)
                        train_step(x_batch, y_batch)
                        current_step = tf.train.global_step(sess, global_step)
                        if current_step % FLAGS.evaluate_every == 0:
                            print("\nEvaluation:")
                            acc = dev_step(x_dev[:10000], y_dev[:10000])
                            print("")
                        if current_step % FLAGS.checkpoint_every == 0:
                            acc = dev_step(x_dev[:10000], y_dev[:10000])
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))

    def eval(self, show_params=False):
        """
        Test datset을 가지고 학습된 모델 성능 평가
        :param show_params: True라면 파라미터 출력
        :return: Test dataset의 모든 softmax 값 리턴
        """

        # show_params가 True라면 하이퍼파라미터 출력
        if show_params:
            print("\nParameters:")
            for attr, value in sorted(FLAGS.__flags.items()):
                print("{}={}".format(attr.upper(), value))
            print("")

        # eval_train이 True라면 학습 데이터와 테스트 데이터 모두 불러오기
        if FLAGS.eval_train:
            x_raw, y_test = data_loader.load_data_and_labels()
            y_test = np.argmax(y_test, axis=1)
        else:
            x_raw, y_test = data_loader.load_dev_data_and_labels()
            y_test = np.argmax(y_test, axis=1)

        x_test = np.array(list(self.vocab_processor.transform(x_raw)))

        print("\nEvaluating...\n")

        # input과 dropout layer tensor 가져오기
        input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # 우리가 알고 싶은 예측 layer tensor
        predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]
        score_layer = self.graph.get_operation_by_name("output/scores").outputs[0]

        # batch size 만큼씩 데이터 서빙
        batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # 최종 결과를 예측할 List
        all_predictions = []
        all_scores = []

        # batch size 만큼 돌면서 결과를 가져옴
        for x_test_batch in batches:
            batch_predictions = self.sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            scores = self.sess.run(score_layer, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            for s in scores:
                all_scores.append(s)

        correct_predictions = float(sum(all_predictions == y_test))
        print("checkpoint - Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
        self.accuracy = correct_predictions / float(len(y_test))
        return all_predictions, all_scores

    def predict(self, product_name, tokenized=False, print_out=True):
        """
        주어진 Raw 문장을 가지고 어느 카테고리인지 예측
        :param product_name: 평가 하고 싶은 product name
        :return: 각 카테고리 별 score ex) [0.13, 0.24, 0....], size: len(category)
        """
        category_db = CategoryManager() # Category DB
        # 주어진 문장을 학습 데이터와 같게 토크나이즈
        if tokenized:
            raw = tokenize(product_name)
        else:
            raw = product_name

        # 이름으로 input layer 가져오기
        input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        # dropout layer 불러오기
        dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Output socre와 preduction layer 불러오기
        predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]
        score_layer = self.graph.get_operation_by_name("output/scores").outputs[0]

        # input data np array 형태로 형변환
        x_test = np.array(list(self.vocab_processor.transform([raw])))

        # dropout 없이 score 얻기
        scores = self.sess.run(score_layer, {input_x: x_test, dropout_keep_prob: 1.0})

        if print_out:
            prediction = self.sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
            # 가장 높은 점수의 category id 불러오기
            cate_id = data_loader.class_labels(prediction)
            # db로부터 불러온 id의 category_name 얻기
            res = category_db.retrieve_cate_name_by_cate_id(int(cate_id[0]))[0]
            print('TEXT CATEGORY : {}'.format(res.get('category_name')))
        return scores

if __name__ == '__main__':
    text_classifier = TextCNNClassifier()
    # DB로부터 읽어와서 text 파일로 저장
    # text_classifier.read_data()


    # Text CNN Classifier 학습
    # text_classifier.train()

    # Test dataset으로 평가
    text_classifier.eval()

    # Raw 문장 주고 어느 Category인지 예측
    scores = text_classifier.predict('블랙 핀 스트라이프 자켓 세일,옴므스타일,간지스타일,남자겨울스타일,남자겨울옷', tokenized=True)

# coding=utf-8
import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, w2v_model, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, sess=None, epsilon=8 / 255, alpha=10 / 255,
            k=5, is_free=False):
        self.l2_reg_lambda = l2_reg_lambda
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.w2v_model = w2v_model
        self.l2_loss = tf.constant(0.0)
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.is_free = is_free
        self.sess = sess
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = tf.Variable(
            tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=0.1),
            name="word_embeddings")  # 1
        embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        self.scores, _, self.l2_loss = self.inference(embedded_chars_expanded, self.l2_loss)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                         labels=self.input_y)
        self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
        # FGSM算法
        # self.attack_with_fgsm(embedded_chars)
        # free算法
        self.attack_with_fgsm(embedded_chars, is_free=self.is_free)
        # pgd算法
        # self.attack_with_pgd(embedded_chars)

    def attack_with_pgd(self, embedded_chars, epsilon=8 / 255, alpha=10 / 255, k=5):
        for i in range(k):
            if i == 0:
                self.gradient, = tf.gradients(self.loss, embedded_chars)
                self.delta = 1 * alpha * tf.sign(self.gradient)
                self.embedded_chars = embedded_chars + tf.clip_by_value(self.delta, -epsilon, epsilon)
            else:
                self.gradient, = tf.gradients(self.loss, self.embedded_chars)
                self.delta = self.delta + 1 * alpha * tf.sign(self.gradient)
                self.embedded_chars = self.embedded_chars + tf.clip_by_value(self.delta, - epsilon, epsilon)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.scores, self.predictions, self.l2_loss = self.inference(self.embedded_chars_expanded,
                                                                         self.l2_loss, reuse=tf.AUTO_REUSE)
            with tf.name_scope("loss"):
                losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses1) + self.l2_reg_lambda * self.l2_loss
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def attack_with_fgsm(self, embedded_chars, is_free=False, epsilon=8 / 255, alpha=10 / 255):
        gradient, = tf.gradients(self.loss, embedded_chars)  # 这里是增加攻击
        delta = 1 * alpha * tf.sign(gradient)  # 扰动
        if is_free:
            self.embedded_chars = embedded_chars + tf.clip_by_value(delta, -epsilon, epsilon)
        else:
            self.embedded_chars = embedded_chars + delta
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        self.scores, self.predictions, self.l2_loss = self.inference(self.embedded_chars_expanded, self.l2_loss,
                                                                     reuse=tf.AUTO_REUSE)
        with tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses1) + self.l2_reg_lambda * self.l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def inference(self, embedded_chars_expanded, l2_loss, reuse=None):
        h_drop = self.cnn_single_layer(embedded_chars_expanded, reuse)
        with tf.variable_scope("output", reuse=reuse):
            W = tf.get_variable(
                "W",
                shape=[self.num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[self.num_classes], initializer=tf.constant_initializer(0.1))
            l2_loss += tf.nn.l2_loss(W)
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            predictions = tf.argmax(scores, 1, name="predictions")
            return scores, predictions, l2_loss

    def cnn_single_layer(self, embedded_chars_expanded, reuse):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=reuse):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.get_variable("W", shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    dtype=tf.float32)
                b = tf.get_variable("b", initializer=tf.constant_initializer(0.1), shape=[self.num_filters],
                                    dtype=tf.float32)
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat

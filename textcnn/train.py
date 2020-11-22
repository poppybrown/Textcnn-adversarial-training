#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow_core.contrib import learn

from data_input_helper import get_text_idx, load_data_and_labels, batch_iter
from textcnn.text_cnn import TextCNN

tf.flags.DEFINE_float("dev_sample_percentage", .15, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "../data/cutclean_label_corpus10000.txt",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("train_label_data_file", "", "Data source for the label data.")
tf.flags.DEFINE_string("w2v_file", "../data/vectors.bin", "w2v_file path")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("num_class", 2, "Number of class (default: 2)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def load_data(w2v_model):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    print("Loading data...")
    x_text, y = load_data_and_labels(FLAGS.train_data_file)

    max_document_length = max([len(x.split(" ")) for x in x_text])  # 文本最长长度
    print('len(x) = ', len(x_text), ' ', len(y))
    print(' max_document_length = ', max_document_length)
    if (w2v_model is None):
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        vocab_size = len(vocab_processor.vocabulary_)

        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(int(time.time()))))
        vocab_processor.save("vocab.txt")
        print('save vocab.txt')
    else:
        x = get_text_idx(x_text, w2v_model.vocab_hash, max_document_length)
        vocab_size = len(w2v_model.vocab_hash)
        print('use w2v .bin')

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    return x_train, x_dev, y_train, y_dev, vocab_size


def train(w2v_model, epsilon=8 / 255, alpha=10 / 255, K=5, is_free=False):
    # Training
    # ==================================================
    x_train, x_dev, y_train, y_dev, vocab_size = load_data(w2v_model)
    # fgsm = FGSM()
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                w2v_model,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                sess=sess)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            # train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy, scores, l2_loss, predictions = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.scores,
                     cnn.l2_loss, cnn.predictions],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                p, r, micro_p, micro_r, micro_f1, macro_f1, accuracy1 = fastF1(np.argmax(y_batch, axis=1), predictions,
                                                                               FLAGS.num_class)
                print(
                    "train {}: step {}, loss {:g}, P: {:.2f}%, R: {:.2f}%, micro_p: {:.2f}%, micro_r: {:.2f}%,Micro_f1: {:.2f}%, Macro_f1: {:.2f}%, Accuracy: {:.2f}%".format(
                        time_str, step, loss, p, r, micro_p, micro_r, micro_f1, macro_f1, accuracy1))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, predictions = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()

                # p, r, micro_p, micro_r, micro_f1, macro_f1, accuracy1 = fastF1(np.argmax(y_batch, axis=1), predictions,
                #                                                                2)
                print(
                    "test {}: step {}, loss {:g}".format(time_str, step, loss))
                if writer:
                    writer.add_summary(summaries, step)
                return predictions

            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            num_batches_per_epoch = int((len(list(zip(x_train, y_train))) - 1) / FLAGS.batch_size) + 1

            def dev_test():
                batches_dev = batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                prediction_all = []
                y_true_all = []
                for batch_dev in batches_dev:
                    x_batch_dev, y_batch_dev = zip(*batch_dev)
                    predictions = dev_step(x_batch_dev, y_batch_dev, writer=dev_summary_writer)
                    prediction_all.extend(predictions)
                    y_true_all.extend(np.argmax(y_batch_dev, axis=1).tolist())
                p, r, micro_p, micro_r, micro_f1, macro_f1, accuracy1 = fastF1(y_true_all, prediction_all,
                                                                               FLAGS.num_class)
                print(
                    "test P: {:.2f}%, R: {:.2f}%, micro_p: {:.2f}%, micro_r: {:.2f}%,Micro_f1: {:.2f}%, Macro_f1: {:.2f}%, Accuracy: {:.2f}%".format(
                        p, r, micro_p, micro_r, micro_f1, macro_f1, accuracy1))
                # dev_step(x_dev, y_dev, writer=dev_summary_writer)

            # Training loop. For each batch...
            num_batches_per_epoch_ = num_batches_per_epoch
            if is_free:
                num_batches_per_epoch_ = int(num_batches_per_epoch / K)
            count_num = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                if count_num == num_batches_per_epoch_:
                    break
                if is_free:
                    for i in range(K):
                        train_step(x_batch, y_batch)
                else:
                    train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                # Training loop. For each batch...
                # 每50步一次
                if current_step % FLAGS.evaluate_every == 0 and current_step > 0:
                    print("\nEvaluation:")
                    dev_test()

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if is_free:
                    count_num += 1


def fastF1(result, predict, num_classes: int):
    ''' f1 score '''
    true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
    total_list = []
    for trueValue in range(num_classes):
        trueNum, recallNum, precisionNum = 0, 0, 0
        for index, values in enumerate(result):
            if values == trueValue:
                recallNum += 1
                if values == predict[index]:
                    trueNum += 1
            if predict[index] == trueValue:
                precisionNum += 1
        R = trueNum / recallNum if recallNum else 0
        P = trueNum / precisionNum if precisionNum else 0
        true_total += trueNum
        r_total += recallNum
        p_total += precisionNum
        p += P
        r += R
        f1 = (2 * P * R) / (P + R) if (P + R) else 0
        total_list.append([P, R, f1])
    p, r = np.array([p, r]) / num_classes
    micro_r, micro_p = true_total / np.array([r_total, p_total])
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) else 0
    accuracy = true_total / len(result)
    # print(
    #     'P: {:.2f}%, R: {:.2f}%, micro_p: {:.2f}%, micro_r: {:.2f}%,Micro_f1: {:.2f}%, Macro_f1: {:.2f}%, Accuracy: {:.2f}%'.format(
    #         p * 100, r * 100, micro_p * 100, micro_r * 100, micro_f1 * 100, macro_f1 * 100, accuracy * 100))
    return p * 100, r * 100, micro_p * 100, micro_r * 100, micro_f1 * 100, macro_f1 * 100, accuracy * 100


if __name__ == "__main__":
    # train(None)
    ## free
    train(None, is_free=True)

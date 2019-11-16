import os
import math
import numpy as np
import tensorflow as tf

NUM_PARALLEL_EXEC_UNITS = 32

sess_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                                       inter_op_parallelism_threads=2,
                                       allow_soft_placement=True,
                                       device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"] = "granularity=fine,noverbose,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"


class DeepRNN:
    def __init__(self, lstm_size, lstm_layers, start_lr, embed_size, drop_rate,
                 max_sent_length, vocab_size, decay_rate, decay_steps, fc_units, multiple_fc,
                 num_classes, save_path, pre_trained=False, embed_matrix: np.ndarray=None):
        tf.compat.v1.reset_default_graph()
        self._lstm_size = lstm_size
        self._lstm_layers = lstm_layers
        self._start_lr = start_lr
        self._embed_size = embed_size
        self._drop_rate = drop_rate
        self._sent_length = max_sent_length
        self._vocab_size = vocab_size
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        self.fc_units = fc_units
        self.multiple_fc = multiple_fc
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.embed_matrix = embed_matrix
        self.save_path = save_path
        self.__trained_embed_matrix = None

        assert embed_matrix is not None if self.pre_trained else True, "Supply a pre-trained embeddings' matrix"

        # --- create placeholders for inputs and targets to feed into graph
        self._inputs = tf.compat.v1.placeholder(tf.int32, shape=[None, self._sent_length], name='inputs')
        self._embedding_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[self._vocab_size, self._embed_size],
                                                     name='embedding_placeholder')
        self._labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='labels')
        self._seqlens = tf.reduce_sum(tf.sign(self._inputs), 1)
        self._keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        self._batch_size = tf.shape(self._inputs)[0]  # an adaptive batch size

        # --- load or randomly initialize the embeddings
        with tf.compat.v1.variable_scope("inference", reuse=tf.compat.v1.AUTO_REUSE):
            if self.pre_trained:
                self._embeddings = tf.compat.v1.get_variable('embeddings',
                                                             initializer=tf.constant(0.0, shape=[self._vocab_size,
                                                                                                 self._embed_size]),
                                                             trainable=True)
                self.embedding_init = self._embeddings.assign(self._embedding_placeholder)
                self.embed = tf.nn.embedding_lookup(self._embeddings, self._inputs)
            else:
                self._embeddings = tf.compat.v1.get_variable('embeddings',
                                                             initializer=tf.random.uniform([self._vocab_size,
                                                                                            self._embed_size],
                                                                                           -0.1, 0.1), trainable=True)
                self.embed = tf.nn.embedding_lookup(self._embeddings, self._inputs)
            self.softmax_w = tf.compat.v1.get_variable('softmax_w',
                                                       initializer=tf.random.truncated_normal([self.fc_units, self.num_classes],
                                                                                              mean=0, stddev=0.01))
            self.softmax_b = tf.compat.v1.get_variable('softmax_b', initializer=tf.random.uniform([self.num_classes]))

        # --- build the RNN layers, get RNN outputs (states)
        with tf.compat.v1.variable_scope("rnn_outputs", reuse=tf.compat.v1.AUTO_REUSE):
            stacked_rnn = list()
            for i in range(self._lstm_layers):
                stacked_rnn.append(self._lstm_cell())
            self._cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
            self._initial_state = self._cell.zero_state(self._batch_size, tf.float32)
            _, self._final_states = tf.nn.dynamic_rnn(
                self._cell,
                self.embed,
                sequence_length=self._seqlens,
                initial_state=self._initial_state,
                dtype=tf.float32
            )

        # --- create the fully connected layers
        with tf.compat.v1.variable_scope("fully_connected", reuse=tf.compat.v1.AUTO_REUSE):
            # self.final_states.h
            self._dense = tf.contrib.layers.fully_connected(self._final_states[self._lstm_layers - 1].h,
                                                            num_outputs=self.fc_units,
                                                            activation_fn=tf.nn.relu,
                                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                            biases_initializer=tf.zeros_initializer())
            self._dense = tf.contrib.layers.dropout(self._dense, self._drop_rate)

            # --- optionally use a second fully connected layer
            if self.multiple_fc:
                self._dense = tf.contrib.layers.fully_connected(self._dense,
                                                                num_outputs=self.fc_units,
                                                                activation_fn=tf.nn.relu,
                                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                                biases_initializer=tf.zeros_initializer())
                self._dense = tf.contrib.layers.dropout(self._dense, self._drop_rate)

        # --- set up the cost operation
        with tf.compat.v1.variable_scope("cost", reuse=tf.compat.v1.AUTO_REUSE):
            self._final_output = tf.compat.v1.nn.xw_plus_b(self._dense, self.softmax_w, self.softmax_b)
            self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._final_output,
                                                                                   labels=self._labels))
        # --- produce predictions
        with tf.compat.v1.variable_scope("predictions", reuse=tf.compat.v1.AUTO_REUSE):
            self._predictions = tf.nn.softmax(self._final_output)

        # --- train the model
        with tf.compat.v1.variable_scope('train', reuse=tf.compat.v1.AUTO_REUSE):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.compat.v1.train.exponential_decay(self._start_lr, global_step,
                                                                 self._decay_steps, self._decay_rate,
                                                                 staircase=True)
            self._optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate)
            gradients, variables = zip(*self._optimizer.compute_gradients(self._cost))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self._train_op = self._optimizer.apply_gradients(zip(gradients, variables),
                                                             global_step=global_step)

        # --- determine the accuracy
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self._labels, 1), tf.argmax(self._final_output, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def _lstm_cell(self):
        # --- basic LSTM cell
        lstm = tf.nn.rnn_cell.LSTMCell(self._lstm_size, reuse=tf.compat.v1.get_variable_scope().reuse)
        # --- add dropout to the cell
        return tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self._drop_rate)

    @staticmethod
    def _get_random_mini_batches(x, y=None, batch_size=64, shuffle=True):
        """
        Returns a generator of random minibatches from (x, y)

        Arguments:
        x       -- input data of sequences
        y       -- true "label" vector in one-hot form

        Returns:
        a generator of minibatches
        """
        m = x.shape[0]  # number of training examples
        # --- shuffle (x, y)
        if shuffle:
            permutation = list(np.random.permutation(m))
            shuffled_x = x[permutation, :]
            if y is not None:
                shuffled_y = y[permutation, :]
        else:
            shuffled_x = x
            if y is not None:
                shuffled_y = y
        # --- partition (shuffled_x, shuffled_y) without the end case.
        num_complete_minibatches = int(math.floor(m / batch_size))
        if y is not None:
            for k in range(0, num_complete_minibatches):
                minibatch_x = shuffled_x[k * batch_size: (k + 1) * batch_size, :]
                minibatch_y = shuffled_y[k * batch_size: (k + 1) * batch_size, :]
                yield minibatch_x, minibatch_y
            # --- handling the end case (last minibatch < batch_size)
            if m % batch_size != 0:
                minibatch_x = shuffled_x[batch_size * num_complete_minibatches:, :]
                minibatch_y = shuffled_y[batch_size * num_complete_minibatches:, :]
                yield minibatch_x, minibatch_y
        else:
            for k in range(0, num_complete_minibatches):
                minibatch_x = shuffled_x[k * batch_size: (k + 1) * batch_size, :]
                yield minibatch_x
            if m % batch_size != 0:
                minibatch_x = shuffled_x[batch_size * num_complete_minibatches:, :]
                yield minibatch_x

    @staticmethod
    def _get_sentence_batch(x, y, batch_size):
        """Get a random train/test batch"""
        instance_idx = list(range(len(x)))
        np.random.shuffle(instance_idx)
        batch = instance_idx[:batch_size]
        x_batch = x[batch, :]
        y_batch = y[batch, :]
        return x_batch, y_batch

    def fit(self, x_train, y_train, x_valid, y_valid, batch_size, epochs=10):
        """Fit the model using minibatch learning.
           Run through ALL minibatches in an epoch
        """
        # --- define a saver to save variables
        saver = tf.compat.v1.train.Saver()
        # --- run a TensorFlow session
        with tf.compat.v1.Session(config=sess_config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            if self.pre_trained:
                sess.run(self.embedding_init, feed_dict={self._embedding_placeholder: self.embed_matrix})
            best_validation_loss = 100000
            for epoch in range(epochs):
                # ----- Train run
                train_losses = list()
                train_accs = list()
                for x_batch_train, y_batch_train in self._get_random_mini_batches(x_train, y=y_train,
                                                                                  batch_size=batch_size):
                    data_dict = {self._inputs: x_batch_train,
                                 self._labels: y_batch_train,
                                 self._keep_prob: self._drop_rate}
                    train_loss, train_acc, _ = sess.run([self._cost, self._accuracy, self._train_op],
                                                        feed_dict=data_dict)
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                # --- get average loss and accuracy over one epoch
                avg_train_loss = np.mean(train_losses)
                avg_train_acc = np.mean(train_accs)

                # ----- Validation run
                valid_losses = list()
                valid_accs = list()
                for x_batch_test, y_batch_test in self._get_random_mini_batches(x_valid, y=y_valid,
                                                                                batch_size=batch_size):
                    data_dict = {self._inputs: x_batch_test, self._labels: y_batch_test,
                                 self._keep_prob: 1}
                    valid_loss, valid_acc = sess.run([self._cost, self._accuracy],
                                                     feed_dict=data_dict)
                    valid_losses.append(valid_loss)
                    valid_accs.append(valid_acc)
                # --- get average loss and accuracy over one epoch
                avg_valid_loss = np.mean(valid_losses)
                avg_valid_acc = np.mean(valid_accs)
                if avg_valid_loss < best_validation_loss:
                    # -- update the best-achieved validation loss
                    self.__trained_embed_matrix = sess.run(self._embeddings)
                    best_validation_loss = avg_valid_loss
                    # --- save session if improved validation accuracy achieved
                    saver.save(sess=sess, save_path=self.save_path)
                    print(
                        f"\nEpoch: {epoch + 1}/{epochs}; train loss: {avg_train_loss}; train acc: {avg_train_acc}; "
                        f"validation loss: {avg_valid_loss}; validation acc: {avg_valid_acc}, improvement found!")
                else:
                    print(
                        f"\nEpoch: {epoch + 1}/{epochs}; train loss: {avg_train_loss}; train acc: {avg_train_acc}; "
                        f"validation loss: {avg_valid_loss}; validation acc: {avg_valid_acc}, no improvement found...")

    def predict(self, x_test, y_test=None, batch_size=64, shuffle=False):
        # --- test run and predictions
        saver = tf.compat.v1.train.Saver()
        test_accs = list()
        all_predictions = list()
        all_labels = list()
        if y_test is not None:
            with tf.compat.v1.Session(config=sess_config) as sess_test:
                saver.restore(sess_test, self.save_path)
                for x_batch_test, y_batch_test, in self._get_random_mini_batches(x_test, y=y_test, batch_size=batch_size,
                                                                                 shuffle=shuffle):
                    test_acc, test_predictions = sess_test.run([self._accuracy, self._predictions],
                                                               feed_dict={self._inputs: x_batch_test,
                                                                          self._labels: y_batch_test,
                                                                          self._keep_prob: 1})
                    test_accs.append(test_acc)
                    all_predictions.append(test_predictions)
                    all_labels.append(y_batch_test)
                all_predictions = np.concatenate(all_predictions, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
            print("Test accuracy: {:.3f}".format(np.mean(test_accs)))
            return all_labels, all_predictions
        else:
            with tf.compat.v1.Session(config=sess_config) as sess_test:
                saver.restore(sess_test, self.save_path)
                for x_batch_test in self._get_random_mini_batches(x_test, batch_size=batch_size, shuffle=shuffle):
                    test_predictions = sess_test.run(self._predictions, feed_dict={self._inputs: x_batch_test,
                                                                                   self._keep_prob: 1})
                    all_predictions.append(test_predictions)
                all_predictions = np.concatenate(all_predictions, axis=0)
            return all_predictions

    @property
    def trained_embed_matrix(self):
        return self.__trained_embed_matrix

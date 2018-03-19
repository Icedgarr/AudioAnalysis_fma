import tensorflow as tf
from tensorflow.python.ops import init_ops


class DieleNet:
    def __init__(self, input_shape, classes=8):
        self.input_shape = input_shape
        self.train = True
        self.classes = classes

    def build_graph(self):
        tf.reset_default_graph()
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.create_ph()
        self.build_conv_layers()
        self.build_glob_temp_pool()
        self.build_dense_layers()
        self.build_output_layer()
        self.build_loss()

    def create_ph(self):
        with tf.name_scope('Data'):
            self.x_ph = tf.placeholder(tf.float32, shape=(None, self.input_shape[0],
                                                          self.input_shape[1]), name='mel')
            self.y_ph = tf.placeholder(tf.int32, shape=(None, self.classes), name='genre')

    def build_conv_layers(self):

        with tf.name_scope('ConvLayers'):
            self.conv1 = tf.layers.conv1d(self.x_ph, filters=256, kernel_size=4, strides=1,
                              padding='valid', data_format='channels_last',
                              dilation_rate=1, activation=tf.nn.relu, use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),#init_ops.glorot_normal_initializer()
                              kernel_regularizer=None, bias_initializer=init_ops.zeros_initializer(),
                              bias_regularizer=None, activity_regularizer=None,
                              kernel_constraint=None, bias_constraint=None,
                              trainable=True, name='conv1', reuse=None)
            self.max_pool1 = tf.layers.max_pooling1d(self.conv1, pool_size=4, strides=4,
                                                     padding='valid', data_format='channels_last',
                                                     name='pool1')
            self.conv2 = tf.layers.conv1d(self.max_pool1, filters=256, kernel_size=4, strides=1,
                                          padding='valid', data_format='channels_last',
                                          dilation_rate=1, activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          kernel_regularizer=None, bias_initializer=init_ops.zeros_initializer(),
                                          bias_regularizer=None, activity_regularizer=None,
                                          kernel_constraint=None, bias_constraint=None,
                                          trainable=True, name='conv2', reuse=None)

            self.max_pool2 = tf.layers.max_pooling1d(self.conv2, pool_size=2, strides=2,
                                                     padding='valid', data_format='channels_last',
                                                     name='pool2')
            self.conv3 = tf.layers.conv1d(self.max_pool2, filters=512, kernel_size=4, strides=1,
                                          padding='valid', data_format='channels_last',
                                          dilation_rate=1, activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          kernel_regularizer=None, bias_initializer=init_ops.zeros_initializer(),
                                          bias_regularizer=None, activity_regularizer=None,
                                          kernel_constraint=None, bias_constraint=None,
                                          trainable=True, name='conv3', reuse=None)

    def build_glob_temp_pool(self):
        with tf.name_scope('Global_temporal_pooling'):
            self.glob_maxpool = tf.reduce_max(self.conv3, axis=0, keep_dims=False,
                                              name='global_max_pool')
            self.glob_meanpool = tf.reduce_mean(self.conv3, axis=0, keep_dims=False,
                                                name='global_mean_pool')
            self.glob_logsum = tf.reduce_logsumexp(self.conv3, axis=0, keep_dims=False,
                                                   name='global_logsum_pool')
            self.glob_L2pool = tf.sqrt(tf.reduce_sum(tf.square(self.conv3), axis=0,
                                                     keep_dims=False), name='global_L2_pool')

            self.glob_temp_pool = tf.concat([self.glob_maxpool, self.glob_meanpool,
                                             self.glob_L2pool, self.glob_logsum],
                                            axis=0, name='global_temporal_pool')

    def build_dense_layers(self):
        with tf.name_scope('DenseLayers'):
            self.dense1 = tf.layers.dense(self.glob_temp_pool, units=2048,
                                          activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=init_ops.zeros_initializer(),
                                          kernel_regularizer=None, bias_regularizer=None,
                                          activity_regularizer=None, kernel_constraint=None,
                                          bias_constraint=None, trainable=True, name='dense1',
                                          reuse=None)
            self.dropout1 = tf.layers.dropout(self.dense1, rate=0.5, noise_shape=None, seed=None,
                                              training=self.train, name='dropout1')
            self.dense2 = tf.layers.dense(self.dropout1, units=2048,
                                          activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=init_ops.zeros_initializer(),
                                          kernel_regularizer=None, bias_regularizer=None,
                                          activity_regularizer=None, kernel_constraint=None,
                                          bias_constraint=None, trainable=True, name='dense2',
                                          reuse=None)
            self.dropout2 = tf.layers.dropout(self.dense2, rate=0.5, noise_shape=None, seed=None,
                                              training=self.train, name='dropout2')

    def build_output_layer(self):
        with tf.name_scope('Output'):
            self.output = tf.layers.dense(self.dropout2, units=self.classes,
                                          activation=tf.nn.softmax, use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=init_ops.zeros_initializer(),
                                          kernel_regularizer=None, bias_regularizer=None,
                                          activity_regularizer=None, kernel_constraint=None,
                                          bias_constraint=None, trainable=True, name='output',
                                          reuse=None)

        with tf.name_scope('Prediction'):
            self.labels = tf.argmax(self.y_ph, axis=1)
            self.predictions = {'classes': tf.argmax(self.output, dimension=1),
                                'probabilities': tf.nn.softmax(self.output)}

            self.accuracy = tf.metrics.accuracy(labels=self.labels,
                                                predictions=self.predictions['classes'],
                                                name='accuracy')

    def build_loss(self, lr=0.0001, beta1=0.9, beta2=0.999, epsilon=10**(-8)):
        with tf.name_scope('Loss'):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output,
                                                                       labels=self.labels,
                                                                       name='Loss')
            self.loss_op = tf.reduce_mean(self.loss)
            print(tf.shape(self.loss_op))
            print(tf.shape(self.labels))
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1,
                                                    beta2=beta2, epsilon=epsilon,
                                                    use_locking=False, name='Adam')
            self.minimizer = self.optimizer.minimize(loss=self.loss_op, global_step=self.global_step,
                                                     var_list=None, name='Minimizer')

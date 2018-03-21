import unittest
import xmlrunner
import numpy as np
import tensorflow as tf

from deep_music.modeling.DieleNet import DieleNet


class TestDieleNet(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_glob_maxpool_keeps_first_dimension(self):
        input = np.ones((5, 647, 256))

        cnn = DieleNet(input_shape=input[0].shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pool = sess.run(cnn.glob_maxpool, feed_dict={cnn.x_ph: input})

        self.assertTrue(pool.shape[0] == 5)

    def test_glob_meanpool_keeps_first_dimension(self):
        input = np.ones((5, 647, 256))

        cnn = DieleNet(input_shape=input[0].shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pool = sess.run(cnn.glob_meanpool, feed_dict={cnn.x_ph: input})

        self.assertTrue(pool.shape[0] == 5)

    def test_glob_logsum_keeps_first_dimension(self):
        input = np.ones((5, 647, 256))

        cnn = DieleNet(input_shape=input[0].shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pool = sess.run(cnn.glob_logsum, feed_dict={cnn.x_ph: input})

        self.assertTrue(pool.shape[0] == 5)

    def test_glob_L2pool_keeps_first_dimension(self):
        input = np.ones((5, 647, 256))

        cnn = DieleNet(input_shape=input[0].shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pool = sess.run(cnn.glob_L2pool, feed_dict={cnn.x_ph: input})

        self.assertTrue(pool.shape[0] == 5)

    def test_glob_temp_pool_keeps_first_dimension(self):
        input = np.ones((5, 647, 256))

        cnn = DieleNet(input_shape=input[0].shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pool = sess.run(cnn.glob_temp_pool, feed_dict={cnn.x_ph: input})

        self.assertTrue(pool.shape[0] == 5)

    def test_output_keeps_first_last_dimensions(self):
        input = np.ones((5, 647, 256))

        cnn = DieleNet(input_shape=input[0].shape, classes=4)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(cnn.output, feed_dict={cnn.x_ph: input})

        self.assertTrue(output.shape[0] == 5)
        self.assertTrue(output.shape[1] == 4)

    def test_pred_classes_labels_same_dimension(self):
        input = np.ones((5, 647, 256))
        output = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                           [1, 0, 0, 0], [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        cnn = DieleNet(input_shape=input[0].shape, classes=4)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            classes, labels = sess.run([cnn.predictions['classes'],
                                        cnn.labels], feed_dict={cnn.x_ph: input,
                                                                cnn.y_ph: output})

        self.assertTrue(classes.shape == labels.shape)




if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output="./python_unittests_xml"))


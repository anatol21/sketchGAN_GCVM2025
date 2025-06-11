import unittest

try:
    import tensorflow as tf
except Exception:
    tf = None


def require_tf(func):
    def wrapper(*args, **kwargs):
        if tf is None:
            raise unittest.SkipTest('TensorFlow not available')
        return func(*args, **kwargs)
    return wrapper


def make_dummy_dataset(batch_size):
    x = tf.zeros([batch_size, 256, 256, 1])
    y = tf.zeros([batch_size, 256, 256, 3])
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)


def simple_generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])
    outputs = tf.keras.layers.Conv2D(3, 1)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def simple_discriminator():
    inp = tf.keras.layers.Input(shape=[256, 256, 1])
    tar = tf.keras.layers.Input(shape=[256, 256, 3])
    x = tf.keras.layers.concatenate([inp, tar])
    x = tf.keras.layers.Conv2D(64, 4, strides=2)(x)
    out = tf.keras.layers.Conv2D(1, 4)(x)
    return tf.keras.Model(inputs=[inp, tar], outputs=out)


class TestDataset(unittest.TestCase):
    @require_tf
    def test_shapes(self):
        ds = make_dummy_dataset(4)
        edge, photo = next(iter(ds))
        self.assertEqual(edge.shape, (4, 256, 256, 1))
        self.assertEqual(photo.shape, (4, 256, 256, 3))


class TestModels(unittest.TestCase):
    @require_tf
    def test_generator_shape(self):
        g = simple_generator()
        out = g(tf.zeros([1, 256, 256, 1]))
        self.assertEqual(out.shape, (1, 256, 256, 3))

    @require_tf
    def test_discriminator_shape(self):
        d = simple_discriminator()
        out = d([tf.zeros([1, 256, 256, 1]), tf.zeros([1, 256, 256, 3])])
        self.assertEqual(out.shape[-1], 1)


if __name__ == '__main__':
    unittest.main()

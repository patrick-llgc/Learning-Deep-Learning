from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Input, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.datasets import cifar10
import pickle
import tensorflow as tf
import keras.backend as K

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10', "Make bottleneck features this for dataset, one of 'cifar10', or 'traffic'")
flags.DEFINE_string('network', 'resnet', "The model to bottleneck, one of 'vgg', 'inception', or 'resnet'")
flags.DEFINE_integer('batch_size', 16, 'The batch size for the generator')

batch_size = FLAGS.batch_size


h, w, ch = 224, 224, 3
if FLAGS.network == 'inception':
    h, w, ch = 299, 299, 3
    from keras.applications.inception_v3 import preprocess_input

img_placeholder = tf.placeholder("uint8", (None, 32, 32, 3))
resize_op = tf.image.resize_images(img_placeholder, (h, w), method=0)


def gen(session, data, labels, batch_size):
    def _f():
        start = 0
        end = start + batch_size
        n = data.shape[0]

        while True:
            X_batch = session.run(resize_op, {img_placeholder: data[start:end]})
            X_batch = preprocess_input(X_batch)
            y_batch = labels[start:end]
            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size

            print(start, end)
            yield (X_batch, y_batch)

    return _f


def create_model():
    input_tensor = Input(shape=(h, w, ch))
    if FLAGS.network == 'vgg':
        model = VGG16(input_tensor=input_tensor, include_top=False)
        x = model.output
        x = AveragePooling2D((7, 7))(x)
        model = Model(model.input, x)
    elif FLAGS.network == 'inception':
        model = InceptionV3(input_tensor=input_tensor, include_top=False)
        x = model.output
        x = AveragePooling2D((8, 8), strides=(8, 8))(x)
        model = Model(model.input, x)
    else:
        model = ResNet50(input_tensor=input_tensor, include_top=False)
    return model


def main(_):

    if FLAGS.dataset == 'cifar10':
        (X_train, y_train), (_, _) = cifar10.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    else:
        with open('data/train.p', mode='rb') as f:
            train = pickle.load(f)
        X_train, X_val, y_train, y_val = train_test_split(train['features'], train['labels'], test_size=0.33, random_state=0)

    train_output_file = "{}_{}_{}.p".format(FLAGS.network, FLAGS.dataset, 'bottleneck_features_train')
    validation_output_file = "{}_{}_{}.p".format(FLAGS.network, FLAGS.dataset, 'bottleneck_features_validation')

    print("Resizing to", (w, h, ch))
    print("Saving to ...")
    print(train_output_file)
    print(validation_output_file)

    with tf.Session() as sess:
        K.set_session(sess)
        K.set_learning_phase(1)

        model = create_model()

        print('Bottleneck training')
        train_gen = gen(sess, X_train, y_train, batch_size)
        bottleneck_features_train = model.predict_generator(train_gen(), X_train.shape[0])
        data = {'features': bottleneck_features_train, 'labels': y_train}
        pickle.dump(data, open(train_output_file, 'wb'))

        print('Bottleneck validation')
        val_gen = gen(sess, X_val, y_val, batch_size)
        bottleneck_features_validation = model.predict_generator(val_gen(), X_val.shape[0])
        data = {'features': bottleneck_features_validation, 'labels': y_val}
        pickle.dump(data, open(validation_output_file, 'wb'))

if __name__ == '__main__':
    tf.app.run()

from tensorflow import keras
import tensorflow as tf


def convert_pretrained_tflite(path, save_path):
    model = keras.models.load_model(path)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile(save_path, 'wb') as f:
        f.write(tflite_model)


def convert_v1_15_tflite(path, save_path):
    # model = keras.models.load_model(path)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model_file(path) #from_keras_model(model)
    # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile(save_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    convert_v1_15_tflite('resnet18.h5', 'resnet18_v1.tflite')

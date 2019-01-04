"""
This code is used to evaluate our trained model against a number of datasets.
"""
import functools
import argparse
import os
import itertools
 
import tensorflow as tf

from model import input_fn, get_model_fn
from dataset_utils import read_label_file

import scipy.misc

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate(model_dir, dataset_dir):
    """
    Begins evaluating the entire architecture.
    """
    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=0,  # Autocompute how many threads to run
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = tf.contrib.learn.RunConfig(
        session_config=sess_config, model_dir=model_dir)

    eval_input_fn = functools.partial(
        input_fn,
        dataset_dir=dataset_dir,
        split_name='validation',
        is_training=False)

    # Get the number of classes from the label file
    labels_to_class_names, num_classes = read_label_file(dataset_dir)

    classifier = tf.estimator.Estimator(
        model_fn=get_model_fn(num_classes),
        config=config)

    # .predict() returns an iterator of dicts;
    y = classifier.predict(input_fn=eval_input_fn)

    num_food_image = {}

    for pred in y:
        predicted_class = labels_to_class_names[int(pred['classes'])]
        food_dir = '../Validations/%s/%s' % (os.path.basename(
            model_dir), predicted_class)

        if not os.path.exists(food_dir):
            os.makedirs(food_dir)

        file_name = os.path.join(food_dir, '%s.png' % num_food_image.get(predicted_class, 1))

        num_food_image[predicted_class] = num_food_image.get(predicted_class, 1) + 1

        scipy.misc.imsave(file_name, pred['features'])


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Evaluate a model against a dataset.')

    PARSER.add_argument('--model',
                        required=True,
                        help='The name of the pre-trained model\'s folder.')

    PARSER.add_argument('--dataset',
                        required=True,
                        help='The folder corresponding to this model\'s dataset.')

    if not os.path.exists(PARSER.parse_args().model):
        raise Exception("Path %s doesn't exist." % PARSER.parse_args().model)

    if not os.path.exists(PARSER.parse_args().dataset):
        raise Exception("Path %s doesn't exist." % PARSER.parse_args().dataset)

    # A (supposed) 5% percent boost in certain GPUs by using faster convolution operations
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    evaluate(PARSER.parse_args().model, PARSER.parse_args().dataset)

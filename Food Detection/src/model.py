import argparse
import functools
import itertools
import os
import six

import tensorflow as tf

import dataset_utils
import utils
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

tf.logging.set_verbosity(tf.logging.INFO)

# =============== CONFIGURATION ===============
IMAGE_SIZE = 299

IMAGES_PER_GPU = 8

GPU_COUNT = 2

BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT

LEARNING_RATE = 0.005

DECAY = 0.9

VALIDATION_STEPS = 50

STEPS_PER_EPOCH = 101000 / BATCH_SIZE

VARIABLE_STRATEGY = 'GPU'

WEIGHT_DECAY = 2e-4

DECAY = 0.9


def tower_fn(is_training, feature, label, num_classes):
    """Build computation tower
    Args:
        is_training: true if is training graph.
        feature: a Tensor.
        label: a Tensor.
    Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """
    with tf.contrib.framework.arg_scope(inception_resnet_v2_arg_scope()):
        logits, endpoints = inception_resnet_v2(feature,
                                                num_classes=num_classes,
                                                is_training=is_training)

    tower_pred = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }

    if label:
        tower_loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits,
            labels=label)

        aux_tower_loss = 0.4 * \
            tf.losses.sparse_softmax_cross_entropy(
                logits=endpoints['AuxLogits'], labels=label)

        tower_loss = tf.reduce_mean(tower_loss + aux_tower_loss)

        model_params = tf.trainable_variables()
        tower_loss += WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in model_params])

        tower_grad = tf.gradients(tower_loss, model_params)

        return tower_loss, zip(tower_grad, model_params), tower_pred

    return None, None, tower_pred


def get_model_fn(num_classes):
    """
    Returns a model function given the number of classes.
    """
    def model_fn(features, labels, mode):
        """Inception_Resnet_V2 model body.
        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
        manages gradient updates.
        Args:
        features: a list of tensors, one for each tower
        labels: a list of tensors, one for each tower
        mode: ModeKeys.TRAIN or EVAL
        Returns:
        A EstimatorSpec object.
        """
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        tower_features = features
        tower_labels = labels
        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
        # on CPU. The exception is Intel MKL on CPU which is optimal with
        # channels_last.
        data_format = None
        if not data_format:
            if GPU_COUNT == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'

        if GPU_COUNT == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = GPU_COUNT
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if VARIABLE_STRATEGY == 'CPU':
                device_setter = utils.local_device_setter(
                    worker_device=worker_device)
            elif VARIABLE_STRATEGY == 'GPU':
                device_setter = utils.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        GPU_COUNT, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope('', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = tower_fn(is_training, tower_features[i],
                                                         tower_labels and tower_labels[i], num_classes)
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            # Only trigger batch_norm moving mean and variance update from
                            # the 1st tower. Ideally, we should grab the updates from all
                            # towers but these stats accumulate extremely fast so we can
                            # ignore the other stats from the other towers without
                            # significant detriment.
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                           name_scope)
        if mode == 'train' or mode == 'eval':
            # Now compute global loss and gradients.
            gradvars = []
            with tf.name_scope('gradient_ing'):
                all_grads = {}
                for grad, var in itertools.chain(*tower_gradvars):
                    if grad is not None:
                        all_grads.setdefault(var, []).append(grad)
                for var, grads in six.iteritems(all_grads):
                    # Average gradients on the same device as the variables
                    # to which they apply.
                    with tf.device(var.device):
                        if len(grads) == 1:
                            avg_grad = grads[0]
                        else:
                            avg_grad = tf.multiply(
                                tf.add_n(grads), 1. / len(grads))
                    gradvars.append((avg_grad, var))

            # Device that runs the ops to apply global gradient updates.
            consolidation_device = '/gpu:0' if VARIABLE_STRATEGY == 'GPU' else '/cpu:0'
            with tf.device(consolidation_device):
                loss = tf.reduce_mean(tower_losses, name='loss')

                examples_sec_hook = utils.ExamplesPerSecondHook(
                    BATCH_SIZE, every_n_steps=10)

                global_step = tf.train.get_global_step()

                learning_rate = tf.constant(LEARNING_RATE)

                tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

                logging_hook = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=100)

                initializer_hook = utils.IteratorInitializerHook()

                train_hooks = [initializer_hook, logging_hook, examples_sec_hook]

                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=LEARNING_RATE, momentum=MOMENTUM)

                # Create single grouped train op
                train_op = [
                    optimizer.apply_gradients(gradvars, global_step=global_step)
                ]
                train_op.extend(update_ops)
                train_op = tf.group(*train_op)

                predictions = {
                    'classes':
                        tf.concat([p['classes'] for p in tower_preds], axis=0),
                    'probabilities':
                        tf.concat([p['probabilities']
                                for p in tower_preds], axis=0)
                }
                stacked_labels = tf.concat(labels, axis=0)
                metrics = {
                    'accuracy':
                        tf.metrics.accuracy(stacked_labels, predictions['classes'])
                }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks,
                eval_metric_ops=metrics)
        else:
            predictions = {
                'classes':
                    tf.concat([p['classes'] for p in tower_preds], axis=0),
                'probabilities':
                    tf.concat([p['probabilities']
                            for p in tower_preds], axis=0),
                'features': tf.concat([feature for feature in features], axis=0)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
    return model_fn


def input_fn(dataset_dir, split_name, is_training):
    """Create input graph for model.
    Args:
      split_name: one of 'train', 'validate' and 'eval'.
    Returns:
      two lists of tensors for features and labels, each of GPU_COUNT length.
    """
    with tf.device('/cpu:0'):
        tfrecord_file_pattern = '%s_%s_*.tfrecord' % (
            os.path.basename(dataset_dir), "%s")

        file_pattern_for_counting = '%s' % (os.path.basename(dataset_dir))

        dataset = dataset_utils.get_split(split_name, dataset_dir,
                                          tfrecord_file_pattern, file_pattern_for_counting)
        image_batch, _, label_batch = dataset_utils.load_batch(
            dataset, BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, is_training)
        if GPU_COUNT <= 1:
            # No GPU available or only 1 GPU.
            return [image_batch], [label_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.
        image_batch = tf.unstack(image_batch, num=BATCH_SIZE, axis=0)
        label_batch = tf.unstack(label_batch, num=BATCH_SIZE, axis=0)
        feature_shards = [[] for i in range(GPU_COUNT)]
        label_shards = [[] for i in range(GPU_COUNT)]
        for i in range(BATCH_SIZE):
            idx = i % GPU_COUNT
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        return feature_shards, label_shards


def get_experiment_fn(dataset_dir):
    """
    Returns the experiment function given a dataset_dir
    """
    # Get the number of classes from the label file
    _, num_classes = dataset_utils.read_label_file(dataset_dir)

    def experiment_fn(run_config, hparams):
        """
        This is a method passed to tf.contrib.learn.learn_runner that will
        return an instance of an Experiment.
        """

        train_input_fn = functools.partial(
            input_fn,
            dataset_dir=dataset_dir,
            split_name='train',
            is_training=True)

        eval_input_fn = functools.partial(
            input_fn,
            dataset_dir=dataset_dir,
            split_name='validation',
            is_training=False)

        classifier = tf.estimator.Estimator(
            model_fn=get_model_fn(num_classes),
            config=run_config)

        return tf.contrib.learn.Experiment(
            classifier,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            train_steps=None,  # Train forever
            eval_steps=VALIDATION_STEPS)
    return experiment_fn


def train(model_dir, dataset_dir):
    """
    Begins training the entire architecture.
    """
    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=0,  # Autocompute how many threads to run
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = tf.contrib.learn.RunConfig(
        session_config=sess_config, model_dir=model_dir)
    tf.contrib.learn.learn_runner.run(
        get_experiment_fn(dataset_dir),
        run_config=config,
        hparams=tf.contrib.training.HParams())


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Train a model against a dataset.')

    PARSER.add_argument('--model', dest='model',
                        required=True,
                        help='The name of the model\'s folder.')

    PARSER.add_argument('--dataset', dest='dataset',
                        required=True,
                        help='The folder corresponding to this model\'s dataset.')

    if not os.path.exists(PARSER.parse_args().model):
        raise Exception("Path %s doesn't exist." % PARSER.parse_args().model)

    if not os.path.exists(PARSER.parse_args().dataset):
        raise Exception("Path %s doesn't exist." % PARSER.parse_args().dataset)

    # A (supposed) 5% percent boost in certain GPUs by using faster convolution operations
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    train(PARSER.parse_args().model, PARSER.parse_args().dataset)

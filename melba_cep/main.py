import os
import numpy as np
import tensorflow.compat.v1 as tf
from collections import OrderedDict
from itertools import chain
import tensorflow_train
import tensorflow_train.utils.tensorflow_util
from tensorflow_train.utils.tensorflow_util import get_reg_loss
from tensorflow_train.utils.data_format import get_batch_channel_image_size
import utils.io.image
import utils.io.landmark
import utils.io.text
from tensorflow_train.data_generator import DataGenerator
from tensorflow_train.train_loop import MainLoopBase
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.landmark_statistics import LandmarkStatistics
from tensorflow_train.utils.heatmap_image_generator import generate_heatmap_target_sigmas_rotation
from tensorflow_train.utils.summary_handler import create_summary_placeholder

from dataset import Dataset
from network import network_scn_mia
from visualization import plot_sigma_error, plot_offsets


class MainLoop(MainLoopBase):
    def __init__(self,
                 network,
                 cv,
                 sigma_regularization,
                 same_sigma=True,
                 heatmap_loss='l2',
                 sigma_loss_power=2.0,
                 output_folder_name='',
                 sigma=3.0,
                 dropout_rate=0.5,
                 use_5_landmarks=False,
                 landmark_sources=None,
                 additional_label_folder='',
                 output_folder_lms=''):
        super().__init__()

        # set the following directories accordingly
        self.base_dataset_folder = '/PATH/TO/DATASET/BASE'  # TODO set path to base_dataset_folder containing: `images`, `original_images` and `setup_ann`
        self.base_output_folder = '/PATH/TO/OUTPUT/BASE'  # TODO set path to some valid output directory

        self.original_image_folder = os.path.join(self.base_dataset_folder, 'original_images')
        self.output_folder = os.path.join(self.base_output_folder, output_folder_lms, f'cv{cv}' if cv >= 0 else 'all', output_folder_name, self.output_folder_timestamp())

        self.network = network
        self.batch_size = 1
        self.max_iter = 40000
        self.dropout_rate = dropout_rate
        self.learning_rate = 0.000001
        self.learning_rates = [self.learning_rate, self.learning_rate * 0.5, self.learning_rate * 0.1]
        self.learning_rate_boundaries = [int(self.max_iter * 0.5), int(self.max_iter * 0.75)]
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = self.test_iter
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.001
        self.invert_transformation = False
        self.cv = cv
        self.use_5_landmarks = use_5_landmarks
        self.landmark_sources = landmark_sources
        original_image_extent = [193.5, 240.0]
        self.image_size = [512, 512]
        self.heatmap_size = [512, 512]
        self.image_spacing = [float(np.max([e / s for e, s in zip(original_image_extent, self.image_size)]))] * 2
        print(self.image_spacing)
        self.image_channels = 1
        self.num_landmarks = 5 if self.use_5_landmarks else 19
        self.heatmap_sigma = sigma
        self.sigma_regularization = sigma_regularization
        self.heatmap_loss = heatmap_loss
        self.sigma_loss_power = sigma_loss_power
        self.same_sigma = same_sigma
        self.sigma_scale = 100.0
        self.data_format = 'channels_first'
        self.save_debug_images = False
        self.tiled_processing = False
        dataset_parameters = {'image_size': self.image_size,
                              'heatmap_size': self.heatmap_size,
                              'image_spacing': self.image_spacing,
                              'num_landmarks': self.num_landmarks,
                              'base_folder': self.base_dataset_folder,
                              'data_format': self.data_format,
                              'save_debug_images': self.save_debug_images,
                              'cv': self.cv,
                              'use_5_landmarks': self.use_5_landmarks,
                              'landmark_sources': self.landmark_sources,
                              'additional_label_folder': additional_label_folder
                              }

        dataset = Dataset(**dataset_parameters)

        self.dataset_train = dataset.dataset_train()
        self.dataset_val = dataset.dataset_val()

        self.network = network
        self.landmark_metrics = ['pe_mean', 'pe_std', 'pe_median', 'or2', 'or25', 'or3', 'or4', 'or10']
        if self.use_5_landmarks:
            self.landmark_metric_prefixes = ['mean'] + self.landmark_sources
        else:
            self.landmark_metric_prefixes = ['mean']
        self.additional_summaries_placeholders_val = OrderedDict([(prefix + '_' + name, create_summary_placeholder(prefix + '_' + name)) for name in self.landmark_metrics for prefix in self.landmark_metric_prefixes])

    def loss_sigmas(self, sigmas, landmarks):
        if self.sigma_regularization == 0.0:
            return tf.constant(0.0, tf.float32)
        return self.sigma_regularization * tf.reduce_sum(tf.pow(tf.reduce_prod(sigmas[None, :], -1) * landmarks[:, :, 0] + 1e-8, self.sigma_loss_power)) / landmarks.get_shape().as_list()[0]

    def loss_function(self, target, prediction):
        if self.heatmap_loss == 'huber':
            delta = 1.0
            error = target - prediction
            abs_error = tf.abs(error)
            quadratic = tf.minimum(abs_error, delta)
            linear = abs_error - quadratic
            return tf.reduce_sum(0.5 * tf.square(quadratic) + delta * linear) / get_batch_channel_image_size(target, self.data_format)[0]
        elif self.heatmap_loss == 'l2':
            return tf.nn.l2_loss(target - prediction) / get_batch_channel_image_size(target, self.data_format)[0]
        return None

    def init_networks(self):
        net = tf.make_template('net', self.network)

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                  ('landmarks', [self.num_landmarks, 3])])
            data_generator_entries_val = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                      ('landmarks', [self.num_landmarks, 3])])
        else:
            data_generator_entries = OrderedDict([('image', list(reversed(self.image_size)) + [self.image_channels]),
                                                  ('landmarks', [self.num_landmarks, 3])])
            data_generator_entries_val = OrderedDict([('image', list(reversed(self.image_size)) + [self.image_channels]),
                                                      ('landmarks', [self.num_landmarks, 3])])

        if self.same_sigma:
            single_sigma = tf.get_variable('sigmas', [self.num_landmarks], initializer=tf.constant_initializer(self.heatmap_sigma))
            self.sigmas = tf.stack([single_sigma, single_sigma], axis=-1)
            self.rotation = tf.constant(0, tf.float32, [self.num_landmarks], 'rotation')
        else:
            self.sigmas = tf.get_variable('sigmas', [self.num_landmarks, 2], initializer=tf.constant_initializer(self.heatmap_sigma))
            self.rotation = tf.get_variable('rotation', [self.num_landmarks], initializer=tf.constant_initializer(0))

        self.train_queue = DataGenerator(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size, n_threads=8)
        placeholders = self.train_queue.dequeue()
        image = placeholders[0]
        target_landmarks = placeholders[1]
        prediction = net(image, num_landmarks=self.num_landmarks, is_training=True, data_format=self.data_format, dropout_rate=self.dropout_rate)
        target_heatmaps = generate_heatmap_target_sigmas_rotation(list(reversed(self.heatmap_size)), target_landmarks, self.sigmas, self.rotation, scale=self.sigma_scale, normalize=True, data_format=self.data_format)
        if self.sigma_regularization == 0.0:
            target_heatmaps = tf.stop_gradient(target_heatmaps)
        loss_sigmas = self.loss_sigmas(self.sigmas, target_landmarks)
        self.loss_reg = get_reg_loss(self.reg_constant)
        self.loss_net = self.loss_function(target_heatmaps, prediction)
        self.loss = self.loss_net + tf.cast(self.loss_reg, tf.float32) + loss_sigmas

        global_step = tf.Variable(self.current_iter, trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rates)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99, use_nesterov=True)
        unclipped_gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        norm = tf.global_norm(unclipped_gradients)
        gradients, _ = tf.clip_by_global_norm(unclipped_gradients, 10000.0)
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
        self.train_losses = OrderedDict([('loss', self.loss_net), ('loss_reg', self.loss_reg), ('loss_sigmas', loss_sigmas), ('norm', norm)])

        # build val graph
        self.val_placeholders = tensorflow_train.utils.tensorflow_util.create_placeholders(data_generator_entries_val, shape_prefix=[1])
        self.image_val = self.val_placeholders['image']
        self.target_landmarks_val = self.val_placeholders['landmarks']
        self.prediction_val = net(self.image_val, num_landmarks=self.num_landmarks, is_training=False, data_format=self.data_format, dropout_rate=self.dropout_rate)
        self.target_heatmaps_val = generate_heatmap_target_sigmas_rotation(list(reversed(self.heatmap_size)), self.target_landmarks_val, self.sigmas, self.rotation, scale=self.sigma_scale, normalize=True, data_format=self.data_format)
        if self.sigma_regularization == 0.0:
            self.target_heatmaps_val = tf.stop_gradient(self.target_heatmaps_val)

        # losses
        self.loss_val = self.loss_function(self.target_heatmaps_val, self.prediction_val)
        self.val_losses = OrderedDict([('loss', self.loss_val), ('loss_reg', self.loss_reg), ('loss_sigmas', tf.constant(0, tf.float32)), ('norm', tf.constant(0, tf.float32))])

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        feed_dict = {self.val_placeholders['image']: np.expand_dims(generators['image'], axis=0),
                     self.val_placeholders['landmarks']: np.expand_dims(generators['landmarks'], axis=0)}

        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.prediction_val, self.target_heatmaps_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(), feed_dict)
        prediction = np.squeeze(run_tuple[0], axis=0)
        target_heatmaps = np.squeeze(run_tuple[1], axis=0)
        image = generators['image']
        transformation = transformations['image']

        return image, prediction, target_heatmaps, transformation

    def finalize_landmark_statistics(self, landmark_statistics, prefix, sigmas):
        pe_mean, pe_std, pe_median = landmark_statistics.get_pe_statistics()
        or2, or25, or3, or4, or10 = landmark_statistics.get_num_outliers([2.0, 2.5, 3.0, 4.0, 10.0], True)
        print(prefix + '_pe', ['{0:.3f}'.format(s) for s in [pe_mean, pe_std, pe_median]])
        print(prefix + '_outliers', ['{0:.3f}'.format(s) for s in [or2, or25, or3, or4, or10]])
        pe_per_landmark = [landmark_statistics.get_pe_statistics([i])[0] for i in range(self.num_landmarks)]
        plot_sigma_error(sigmas, pe_per_landmark, self.output_file_for_current_iteration(prefix + '_sigma_error.pdf'))
        plot_offsets(landmark_statistics.groundtruth_landmarks, landmark_statistics.predicted_landmarks, self.output_file_for_current_iteration(prefix + '_offsets.pdf'), self.original_image_folder)
        overview_string = landmark_statistics.get_overview_string([2, 2.5, 3, 4, 10, 20], 10, 20.0)
        utils.io.text.save_string_txt(overview_string, self.output_file_for_current_iteration(prefix + '_eval.txt'))
        additional_summaries = {prefix + '_pe_mean': pe_mean,
                                prefix + '_pe_std': pe_std,
                                prefix + '_pe_median': pe_median,
                                prefix + '_or2': or2,
                                prefix + '_or25': or25,
                                prefix + '_or3': or3,
                                prefix + '_or4': or4,
                                prefix + '_or10': or10}
        return additional_summaries

    def test(self):
        heatmap_test = HeatmapTest(channel_axis=0, invert_transformation=False)

        if self.use_5_landmarks:
            statistics_entries = ['mean']
            statistics_entries += self.landmark_sources
        else:
            statistics_entries = ['mean']
        landmark_statistics_dict = {x: LandmarkStatistics() for x in statistics_entries}

        landmarks = {}
        for i in range(self.dataset_val.num_entries()):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            reference_image = datasources['image_datasource']
            image, prediction, target_heatmaps, transform = self.test_full_image(dataset_entry)

            utils.io.image.write_multichannel_np(image, self.output_file_for_current_iteration(current_id + '_image.nii.gz'), output_normalization_mode=(-1, 1), image_type=np.uint8, is_single_channel=True, data_format=self.data_format)
            utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(current_id + '_heatmap.nii.gz'), output_normalization_mode=(0, 0.255 * self.sigma_scale), image_type=np.uint8, data_format=self.data_format)
            utils.io.image.write_multichannel_np(target_heatmaps, self.output_file_for_current_iteration(current_id + '_target_heatmap.nii.gz'), output_normalization_mode=(0, 0.255 * self.sigma_scale), image_type=np.uint8, data_format=self.data_format)
            predicted_landmarks = heatmap_test.get_landmarks(prediction, reference_image, output_spacing=self.image_spacing, transformation=transform)
            tensorflow_train.utils.tensorflow_util.print_progress_bar(i, self.dataset_val.num_entries())
            landmarks[current_id] = predicted_landmarks

            for key, landmark_statistics in landmark_statistics_dict.items():
                current_gt_landmarks = datasources[key + '_landmarks_datasource']
                landmark_statistics.add_landmarks(current_id, predicted_landmarks, current_gt_landmarks)

        tensorflow_train.utils.tensorflow_util.print_progress_bar(self.dataset_val.num_entries(), self.dataset_val.num_entries())

        sigmas = self.sess.run(self.sigmas)
        utils.io.text.save_list_csv(sigmas.tolist(), self.output_file_for_current_iteration('sigmas.txt'))
        rotations = self.sess.run(self.rotation)
        utils.io.text.save_list_csv(rotations.tolist(), self.output_file_for_current_iteration('rotations.txt'))
        prod_sigmas = sigmas[:,0] * sigmas[:,1]
        utils.io.text.save_list_csv(prod_sigmas.tolist(), self.output_file_for_current_iteration('sigmas_prod.txt'))

        summaries_list = []
        for key, landmark_statistics in landmark_statistics_dict.items():
            summaries = self.finalize_landmark_statistics(landmark_statistics, key, prod_sigmas)
            summaries_list.append(summaries)

        additional_summaries = OrderedDict(chain(*[x.items() for x in summaries_list]))

        # finalize loss values
        self.val_loss_aggregator.finalize(self.current_iter, additional_summaries)
        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('prediction.csv'))

if __name__ == '__main__':
    tf.disable_eager_execution()  # Note: disabling required for tf1 compatibility

    # True: train on 5 landmarks with 11 annotations available
    # False: train on all 19 landmarks with 2 annotations available (we recommend training using only the junior annotations as discussed in the paper)
    use_5_landmarks = True

    # anisotropic distribution with learned sigmas (proposed)
    same_sigma = False
    sigma_regularization = 5.0

    # # isotropic distribution with learned sigma
    # same_sigma = True
    # sigma_regularization = 5.0

    # # isotropic distribution with fixed sigma
    # same_sigma = True
    # sigma_regularization = 0.0


    if use_5_landmarks:
        landmark_sources = ['junior', 'senior', 'ann1', 'ann2', 'ann3', 'ann4', 'ann5', 'ann6', 'ann7', 'ann8', 'ann9']
        additional_label_folder = '5_landmarks'
        output_folder_lms = '5Lms'
    else:
        landmark_sources = ['junior']  # options (one or multiple): ['junior', 'senior', 'challenge']
        additional_label_folder = 'all_landmarks'
        output_folder_lms = 'allLms'

    for cv in [1, 2, 3, 4]:  # [1, 2, 3, 4, -1]  # cv: -1 is only available when 19 landmarks are used
        output_folder_name = 'same{}_sigma{}'.format(same_sigma, sigma_regularization)
        loop = MainLoop(network=network_scn_mia,
                        cv=cv,
                        sigma_regularization=sigma_regularization,
                        same_sigma=same_sigma,
                        heatmap_loss='l2',
                        sigma_loss_power=1.0,
                        output_folder_name=output_folder_name,
                        sigma=3.0,
                        dropout_rate=0.5,
                        use_5_landmarks=use_5_landmarks,
                        landmark_sources=landmark_sources,
                        additional_label_folder=additional_label_folder,
                        output_folder_lms=output_folder_lms)
        loop.run()

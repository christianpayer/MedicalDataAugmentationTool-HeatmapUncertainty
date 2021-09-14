import os
import numpy as np
import SimpleITK as sitk
from utils.landmark.common import get_mean_landmark_list
from graph.node import LambdaNode
from datasets.graph_dataset import GraphDataset
from datasources.image_datasource import ImageDataSource
from datasources.cached_image_datasource import CachedImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from generators.landmark_generator import LandmarkGenerator
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.spatial import translation, scale, composite, rotation, deformation
from transformations.intensity.sitk.smooth import gaussian


class Dataset(object):
    """
    The dataset that processes files from hand xray dataset.
    """
    def __init__(self,
                 image_size,
                 image_spacing,
                 heatmap_size,
                 num_landmarks,
                 base_folder,
                 cv=-1,
                 image_gaussian_sigma=0.15,
                 use_5_landmarks=False,
                 landmark_sources=None,
                 additional_label_folder='',
                 data_format='channels_first',
                 save_debug_images=False):
        """
        Initializer.
        :param image_size: Network input image size.
        :param heatmap_size: Network output image size.
        :param num_landmarks: The number of landmarks.
        :param sigma: The heatmap sigma.
        :param base_folder: Dataset base folder.
        :param cv: Cross validation index (1, 2, 3).
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.heatmap_size = heatmap_size
        self.downsampling_factor = self.image_size[0] / self.heatmap_size[0]
        self.num_landmarks = num_landmarks
        self.base_folder = base_folder
        self.cv = cv
        self.image_gaussian_sigma = image_gaussian_sigma
        self.use_5_landmarks = use_5_landmarks
        self.additional_label_folder = additional_label_folder

        self.landmark_sources = landmark_sources or ['junior']  # ['junior', 'senior', 'challenge']

        self.data_format = data_format
        self.save_debug_images = save_debug_images
        self.dim = 2
        self.image_base_folder = os.path.join(self.base_folder, 'images')

        self.setup_base_folder = os.path.join(self.base_folder, 'setup_ann', additional_label_folder)

        if cv > 0:
            self.train_id_list_file_name = os.path.join(self.setup_base_folder, 'cv', 'set{}'.format(cv), 'train.txt')
            self.val_id_list_file_name = os.path.join(self.setup_base_folder, 'cv', 'set{}'.format(cv), 'val.txt')
        else:
            self.train_id_list_file_name = os.path.join(self.setup_base_folder, 'train.txt')
            self.val_id_list_file_name = os.path.join(self.setup_base_folder, 'test_all.txt')
        if self.use_5_landmarks:
            self.point_list_file_names = dict([(key, os.path.join(self.setup_base_folder, f'5lms_{key}.csv')) for key in self.landmark_sources])
        else:
            self.point_list_file_names = dict([(key, os.path.join(self.setup_base_folder, f'all_{key}.csv')) for key in self.landmark_sources])

    def image_preprocessing(self, image):
        if self.image_gaussian_sigma > 0.0:
            return gaussian(image, self.image_gaussian_sigma)
        else:
            return image

    def data_sources(self, cached, iterator, image_extension='.nii.gz'):
        """
        Returns the data sources that load data.
        {
        'image_datasource:' ImageDataSource that loads the image files.
        'landmarks_datasource:' LandmarkDataSource that loads the landmark coordinates.
        }
        :param cached: If true, use a CachedImageDataSource instead of an ImageDataSource.
        :param image_extension: The image extension of the input data.
        :return: A dict of data sources.
        """
        if cached:
            image_datasource = CachedImageDataSource(self.image_base_folder,
                                                     '',
                                                     '',
                                                     image_extension,
                                                     preprocessing=self.image_preprocessing,
                                                     set_identity_spacing=False,
                                                     cache_maxsize=16384,
                                                     parents=[iterator],
                                                     name='image_datasource')
        else:
            image_datasource = ImageDataSource(self.image_base_folder,
                                               '',
                                               '',
                                               image_extension,
                                               preprocessing=self.image_preprocessing,
                                               set_identity_spacing=False,
                                               parents=[iterator],
                                               name='image_datasource')
        landmark_datasources = self.landmark_datasources(iterator)
        return {'image_datasource': image_datasource, **landmark_datasources}

    def landmark_datasources(self, iterator):
        landmark_datasources = dict([(key, LandmarkDataSource(self.point_list_file_names[key],
                                                         self.num_landmarks,
                                                         self.dim,
                                                         parents=[iterator],
                                                         name=f'{key}_landmarks_datasource')) for key in self.landmark_sources])
        mean_landmarks_datasource = LambdaNode(lambda *x: get_mean_landmark_list(*x), parents=[landmark_datasources[key] for key in self.landmark_sources], name='mean_landmarks_datasource')
        landmark_datasource = mean_landmarks_datasource
        return {**landmark_datasources,
                'mean_landmarks_datasource': mean_landmarks_datasource,
                'landmark_datasource': landmark_datasource}

    def data_generators(self, data_sources, transformation, image_post_processing_np):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param image_post_processing_np: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        image_generator = ImageGenerator(self.dim,
                                         self.image_size,
                                         self.image_spacing,
                                         post_processing_np=image_post_processing_np,
                                         interpolator='linear',
                                         resample_default_pixel_value=0,
                                         data_format=self.data_format,
                                         resample_sitk_pixel_type=sitk.sitkFloat32,
                                         np_pixel_type=np.float32,
                                         parents=[data_sources['image_datasource'], transformation], name='image')
        if self.downsampling_factor == 1:
            heatmap_post_transformation = None
        else:
            heatmap_post_transformation = scale.Fixed(self.dim, self.downsampling_factor)
        landmark_generator = LandmarkGenerator(self.dim,
                                               self.heatmap_size,
                                               self.image_spacing,
                                               data_format=self.data_format,
                                               post_transformation=heatmap_post_transformation,
                                               min_max_transformation_distance=30,
                                               parents=[data_sources['landmark_datasource'], transformation], name='landmarks')
        return {'image': image_generator,
                'landmarks': landmark_generator}

    def spatial_transformation_augmented(self, datasources):
        """
        The spatial image transformation with random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    translation.Random(self.dim, [10, 10]),
                                    rotation.Random(self.dim, [0.25]),
                                    scale.RandomUniform(self.dim, 0.2),
                                    scale.Random(self.dim, [0.2, 0.2]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                    deformation.Output(self.dim, [5, 5], 10, self.image_size, self.image_spacing)
                                    ],
                                   kwparents={'image': datasources['image_datasource']}, name='image')

    def spatial_transformation(self, datasources):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing)],
                                   kwparents={'image': datasources['image_datasource']}, name='image')

    def intensity_postprocessing_augmented(self, image):
        """
        Intensity postprocessing. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=-128,
                               scale=1/128,
                               random_shift=0.25,
                               random_scale=0.25,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def intensity_postprocessing(self, image):
        """
        Intensity postprocessing.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=-128,
                               scale=1/128,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_id_list_file_name,
                                  random=True,
                                  keys=['image_id'])
        data_sources = self.data_sources(True, iterator)
        image_transformation = self.spatial_transformation_augmented(data_sources)
        data_generators = self.data_generators(data_sources, image_transformation, self.intensity_postprocessing_augmented)
        return GraphDataset(data_generators=list(data_generators.values()),
                            data_sources=list(data_sources.values()),
                            transformations=[image_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        iterator = IdListIterator(self.val_id_list_file_name,
                                  random=False,
                                  keys=['image_id'])
        data_sources = self.data_sources(False, iterator)
        image_transformation = self.spatial_transformation(data_sources)
        data_generators = self.data_generators(data_sources, image_transformation, self.intensity_postprocessing)
        return GraphDataset(data_generators=list(data_generators.values()),
                            data_sources=list(data_sources.values()),
                            transformations=[image_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)



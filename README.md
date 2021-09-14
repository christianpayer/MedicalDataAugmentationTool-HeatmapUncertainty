# Modeling Annotation Uncertainty with Gaussian Heatmaps in Landmark Localization

## Usage
This example implements the landmark localization network of the paper **Modeling Annotation Uncertainty with Gaussian Heatmaps in Landmark Localization** which was used for the inter-observer experiments on the cephalogram dataset with additional annotations.
The implementation for the initial experiments using the hand and cephalogram dataset were originally published for the paper [Integrating Spatial Configuration into Heatmap Regression Based CNNs for Landmark Localization](https://github.com/christianpayer/MedicalDataAugmentationTool-HeatmapRegression) and can be found there.

You need to have the [MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool) framework downloaded and in your `PYTHONPATH` for the scripts to work.
If you have problems/questions/suggestions about the code, write a mail to [Christian Payer](mailto:christian.payer@gmx.net) or [Franz Thaler](mailto:franz.thaler@icg.tugraz.at)!

### Dataset preprocessing
The dataset used in this work is publicly available and can be downloaded from the website: [ISBI 2015 Cephalometric X-ray Image Analysis Challenge](http://www-o.ntust.edu.tw/~cweiwang/ISBI2015/challenge1/).

The script `convert_bmp_to_nii.py` can be used to convert the `.bmp` images to `.nii.gz`.

The annotations including our nine additional annotations for five landmarks and 100 images of the cephalogram dataset can be found in the `setup_ann` directory.

The following structure is expected for the dataset:

    .                             # The `base_dataset_folder` of the dataset
    ├── images                    # Image folder containing all images as `.nii.gz`
    │   ├── 001.nii.gz            
    │   ├── ...                   
    │   └── 400.nii.gz            
    ├── original_images           # Image folder containing all images as `.bmp`
    │   ├── 001.bmp            
    │   ├── ...                   
    │   └── 400.bmp            
    └── setup_ann                 # Setup folder as provided in this repository


### Setup
In `main.py`, please set `self.base_folder`, `self.image_folder` and `self.output_folder`:

- `self.base_dataset_folder`: Base folder of the dataset containing a folder `images`, `original_images` and `setup_ann`. 
- `self.base_output_folder`: Output folder for the experiments, some additional subfolders are generated automatically. Please make sure that the output directory is on a valid disc with sufficient free disc space.


### Train models
Run `main.py` to train the network. 

Adapt parameters in the file to modify the experiment and to define cross validation or the full training/testing.
The bool `use_5_landmarks` can be used to switch between experiments predicting 5 landmarks using the mean of the 11 annotations (junior, senior and the nine annotations acquired in this work) and predicting all 19 landmarks trained using only the junior annotations.

One of the following parameter configurations can be selected for different experiments evaluated in our work:

- Anisotropic distribution with learned sigmas (proposed):
```
same_sigma = False
sigma_regularization = 5.0
```

- Isotropic distribution with learned sigma:
```
same_sigma = True
sigma_regularization = 5.0
```

- Isotropic distribution with fixed sigma:
```
same_sigma = True
sigma_regularization = 0.0
```

## Citation
If you use this code for your research, please cite our paper:

```
@article{Thaler2021,
title   = {Modeling Annotation Uncertainty with Gaussian Heatmaps in Landmark Localization},
author  = {Thaler, Franz and Payer, Christian and Urschler, Martin and {\v{S}}tern, Darko},
journal = {Journal of Machine Learning for Biomedical Imaging},
year    = {2021}
}
```
 

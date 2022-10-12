# DDIR
The source code of "A Deep Discontinuity-Preserving Image Registration Network", which aims to solve the discontinuity-preserving problem in deep learning based image registration networks.

## Contents
- <a href="#Abstract">`Abstract`</a>
- <a href="#Network">`Network`</a>
- <a href="#Repo Contents">`Repo Contents`</a>
- <a href="#Package dependencies">`Package dependencies`</a>
- <a href="#Dataset">`Dataset`</a>
- <a href="#Training">`Training`</a>
- <a href="#Testing">`Testing`</a>
- <a href="#Citation">`Citation`</a>

## Abstract:<a id="Abstract"/>
Image registration aims to establish spatial correspondence across pairs, or groups of images, and is a cornerstone of medical image computing and computer-assisted-interventions. Currently, most deep learning-based registration methods assume that the desired deformation fields are globally smooth and continuous, which is not always valid for real-world scenarios, especially in medical image registration (e.g. cardiac imaging and abdominal imaging). Such a global constraint can lead to artefacts and increased errors at discontinuous tissue interfaces. To tackle this issue, we propose a weakly-supervised Deep Discontinuity-preserving Image Registration network (DDIR), to obtain better registration performance and realistic deformation fields. We demonstrate that our method achieves significant improvements in registration accuracy and predicts more realistic deformations, in registration experiments on cardiac magnetic resonance (MR) images from UK Biobank Imaging Study (UKBB), than state-of-the-art approaches.

## Network:<a id="Network"/>
There are two main components in the DDIR, the Multi-channel Encoder-decoder block and Discontinuity Composition block.
![image](https://github.com/cistib/DDIR/blob/main/fig/DDIR.png)

## Repo Contents:<a id="Repo Contents"/>
This code partially refers to the [voxelmorph](https://github.com/voxelmorph/voxelmorph).

## Package dependencies:<a id="Package dependencies"/>
This repository is based on Python3.6, Tensorflow and Keras.
The versions of the main packages are as follows,
- Tensorflow==1.5.0
- Keras==2.2.4

## Dataset:<a id="Dataset"/>
Our network is trained and tested based on cardiac MR images from UKBB (intra-subject registration). If you want to train DDIR by yourself but have no access to the UKBB, ACDC or M&M dataset could be another option.

## Training:<a id="Training"/>
Use the following command to train the DDIR.
```sh
cd DDIR/
python train.py --data_dir path/to/trainfile/  --gpu 0 --model_dir path/to/model file/
```
where 'data_dir' and 'model_dir' refer to the folder of training data and folder to save the model.

## Testing:<a id="Testing"/>
Use the following command to test the DDIR. Dice score, Hausdorff Distance (HD) and several clinical indices are evaluated in this paper.
```sh
python test.py 0 path/to/model file/  model_name
```
where 'model_name' is the name of a specific model to test.


## Citation:<a id="Citation"/>
If you find this code useful in your research, please consider citing:
```
@inproceedings{chen2021deep,
  title={A Deep Discontinuity-Preserving Image Registration Network},
  author={Chen, Xiang and Xia, Yan and Ravikumar, Nishant and Frangi, Alejandro F},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={46--55},
  year={2021},
  organization={Springer}
}
```

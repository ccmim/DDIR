# DDIR
The source code of "A Deep Discontinuity-Preserving Image Registration Network", which is a deep learning-based image registration network, taking the discontinuity-preserving into consideration.

## Contents
- <a href="#Abstract">`Abstract`</a>
- <a href="#Network">`Network`</a>
- <a href="#Repo Contents">`Repo Contents`</a>
- <a href="#Package dependencies">`Package dependencies`</a>
- <a href="#Dataset">`Dataset`</a>
- <a href="#Training">`Training`</a>
- <a href="#Testing">`Testing`</a>
- <a href="#Demo">`Demo`</a>
- <a href="#Citation">`Citation`</a>

## Abstract:<a id="Abstract"/>
Image registration aims to establish spatial correspondence across pairs, or groups of images, and is a cornerstone of medical image computing and computer-assisted-interventions. Currently, most deep learning-based registration methods assume that the desired deformation fields are globally smooth and continuous, which is not always valid for real-world scenarios, especially in medical image registration (e.g. cardiac imaging and abdominal imaging). Such a global constraint can lead to artefacts and increased errors at discontinuous tissue interfaces. To tackle this issue, we propose a weakly-supervised Deep Discontinuity-preserving Image Registration network (DDIR), to obtain better registration performance and realistic deformation fields. We demonstrate that our method achieves significant improvements in registration accuracy and predicts more realistic deformations, in registration experiments on cardiac magnetic resonance (MR) images from UK Biobank Imaging Study (UKBB), than state-of-the-art approaches.

## Network:<a id="Network"/>
There are two main components in the DDIR, multi-channel encoder-decoder and discontinuity composition block.
![image](https://github.com/cistib/DDIR/blob/main/fig/DDIR.png)

## Repo Contents:<a id="Repo Contents"/>
This code is partially referred to [voxelmorph](https://github.com/voxelmorph/voxelmorph), where the GCN block and the mesh loss are mainly from it.

## Package dependencies:<a id="Package dependencies"/>
This repository is based on Python3.6, Tensorflow and Keras.
The versions of the main packages are as follows,
- Tensorflow==1.5.0
- Keras==2.2.4

## Dataset:<a id="Dataset"/>
Our network is trained and tested based on UKBB cardiac MR images. If you want to train DDIR by yourself but have no access to the UKBB, ACDC dataset could be another option.

## Training:<a id="Training"/>
Use the following command to train the DDIR.
```sh
python train.py --data_dir path/to/trainfile/  --gpu 0 --model_dir path/to/model file/
```

## Testing:<a id="Testing"/>
Use the following command to test the DDIR. Dice score, Hausdorff Distance (HD) and several clinical indices are evaluated in this paper.
```sh
python test.py 0 path/to/model file/  modelname
```

## Demo:<a id="Demo"/>
To reconstruct 3D cardiac mesh with pretrained model from contours.
```sh
python demo.py 0 path/to/model file/  modelname
```

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

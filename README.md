# pytorch_mpiifacegaze
PyTorch MPIIFaceGaze implementation

## Description

This is a PyTorch implementation from the "It's written all over your face" paper [1]. It was based on [2], but some modifications were made in order to better replicate [1] (L1 loss instead of MSE, ADAM instead of SGD, for example).

Also, there's a live demo for the normalization of the whole face as in [3] and [4], for future live experimentations.

## Usage

Currently this only supports training with the MPIIFaceGaze dataset and the face normalization demo. For training your model, you could type something like this: 

''
python main.py --dataset path --batch_size 256 --num_workers 8
''

## License

You should respect the licenses from the MPIIFaceGaze dataset, and of all codes used in this project, as referenced in the next section.

## References
[1] Sugano, Y., Fritz, M., & Andreas Bulling, X. (2017). It's written all over your face: Full-face appearance-based gaze estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 51-60).

[2] ti-gingko's implementation https://github.com/ti-ginkgo/MPIIFaceGaze

[3] Y. Sugano, Y. Matsushita, and Y. Sato. Learning-by-synthesis for appearance-based 3d gaze estimation. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 1821–1828. IEEE, 2014.

[4] Xucong Zhang; Yusuke Sugano; Andreas Bulling. Revisiting Data Normalization for Appearance-Based Gaze Estimation. Proc. International Symposium on Eye Tracking Research and Applications (ETRA), pp. 12:1-12:9, 2018.

# GANimation: Anatomically-aware Facial Animation from a Single Image

## TensorFlow Version
2.4

# Other Requirements
- OS: [Ubuntu](https://ubuntu.com/)
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units)(ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY)<br>
  Follow the commands below to install and set the environment variable of OpenFace
  ```
  git clone https://github.com/TadasBaltrusaitis/OpenFace.git
  cd OpenFace
  bash ./download_models.sh
  sudo bash ./install.sh
  export PATH="`pwd`/build/bin:$PATH"
  ```
- [Face Recognition](https://github.com/ageitgey/face_recognition)
  1. To install Face Recognition, you need to install [dlib](https://github.com/davisking/dlib) first. To install dlib, please refer to the following page.<br>
  https://github.com/ageitgey/face_recognition/blob/master/Dockerfile#L6-L34
  2. Then install Face Recgnition as follows.
  ```
  pip install face_recognition
  ```



## Datasets
[CelebA](https://www.tensorflow.org/datasets/catalog/celeb_a)

## Prepare the Dataset
**coming soon**

## Train
**coming soon**

## Test
**coming soon**

## References
- Paper
  - [GANimation: Anatomically-aware Facial Animation from a Single Image](https://arxiv.org/abs/1807.09251)<br>

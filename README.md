# Monocular Total Capture
Code for CVPR19 paper "Monocular Total Capture: Posing Face, Body and Hands in the Wild"

![Teaser Image](https://xiangdonglai.github.io/MTC_teaser.jpg)

Project website: [<http://domedb.perception.cs.cmu.edu/mtc.html>]

# Dependencies

```
MTC_DIR=~/Projects/MTC && mkdir -p $MTC_DIR
```

#### CUDA
CUDA Driver 550, CUDA toolkit 12.6, cudnn 9.5.1

#### Python
```
pip3 install setuptools \
    wheel \
    tensorflow \
    opencv-python \
    scikit-image \
    Mako \
    matplotlib \
    numpy \
    protobuf
```

#### Ceres Solver
```
sudo apt-get install cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev
```
```
rm -rf $MTC_DIR/eigen && \
cd $MTC_DIR && git clone --branch "3.3.9" https://gitlab.com/libeigen/eigen.git && \
mkdir -p eigen/build && cd eigen/build && cmake .. && sudo make install
```
```
rm -rf $MTC_DIR/ceres-solver && \
cd $MTC_DIR && git clone --branch "1.13.0" https://github.com/ceres-solver/ceres-solver.git && \
mkdir -p ceres-solver/ceres-bin && cd ceres-solver/ceres-bin && cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
make -j`nproc` && sudo make install
```

#### OpenGL
```
sudo apt-get install freeglut3-dev libglew-dev libglm-dev
```

#### libigl
```
rm -rf $MTC_DIR/libigl && \ 
cd $MTC_DIR && git clone --branch "v2.1.0" https://github.com/libigl/libigl.git
```

#### OpenCV
```
rm -rf $MTC_DIR/opencv && \
cd $MTC_DIR && git clone --depth 1 --branch "4.10.0" https://github.com/opencv/opencv && \
git clone --depth 1 --branch "4.10.0" https://github.com/opencv/opencv_contrib && \
mkdir -p opencv/build && cd opencv/build && \
cmake -DOPENCV_EXTRA_MODULES_PATH=$MTC_DIR/opencv_contrib/modules/ \
       -DBUILD_SHARED_LIBS=OFF \
       -DBUILD_TESTS=OFF \
       -DBUILD_PERF_TESTS=OFF \
       -DBUILD_EXAMPLES=OFF \
       -DWITH_OPENEXR=OFF \
       -DWITH_CUDA=ON \
       -DWITH_CUBLAS=ON \
       -DWITH_CUDNN=ON \
       -DOPENCV_DNN_CUDA=ON \
       $MTC_DIR/opencv && \
make -j`nproc` && sudo make install
```

#### OpenPose
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libgflags-dev libgoogle-glog-dev liblmdb-dev
```
```
rm -rf $MTC_DIR/openpose && \
cd $MTC_DIR && git clone --depth 1 --branch "v1.7.0" https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```

Workaround for server connection problem
```
pip install gdown && \
cd $MTC_DIR/openpose && \
gdown --fuzzy https://drive.google.com/file/d/1cqreuG8hSjtGTbtiunQxHWIO9tC8uCaL/view?usp=sharing && \
unzip -o models.zip
```
```
cd $MTC_DIR && mkdir -p openpose/build && cd openpose/build && \
cmake .. -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF && \
make -j`nproc`
```

# Installation

```
rm -rf $MTC_DIR/MonocularTotalCapture && \
cd $MTC_DIR && git clone --depth 1 https://github.com/reenor/MonocularTotalCapture.git && \
cd $MTC_DIR/MonocularTotalCapture && bash download.sh
```

```
cd $MTC_DIR/MonocularTotalCapture/FitAdam && mkdir build && cd build && \
cmake .. && make -j`nproc`
```

# Dependencies
This code is tested on a Ubuntu 16.04 machine with a GTX 1080Ti GPU, with the following dependencies.
1. ffmpeg
2. Python 3.5 (with TensorFlow 1.5.0, OpenCV, Matplotlib, packages installed with pip3)
3. cmake >= 2.8
4. OpenCV 2.4.13 (compiled from source with CUDA 9.0, CUDNN 7.0)
5. Ceres-Solver 1.13.0 (with SuiteSparse)
6. OpenGL, GLUT, GLEW
7. libigl <https://github.com/libigl/libigl>
8. wget
9. OpenPose

# Installation
1. git clone this repository; suppose the main directory is ${ROOT} on your local machine.
2. "cd ${ROOT}"
3. "bash download.sh"
4. git clone OpenPose <https://github.com/CMU-Perceptual-Computing-Lab/openpose> and compile. Suppose the main directory of OpenPose is ${openposeDir}, such that the compiled binary is at ${openposeDir}/build/examples/openpose/openpose.bin
5. Edit ${ROOT}/run_pipeline.sh: set line 13 to you ${openposeDir}
4. Edit ${ROOT}/FitAdam/CMakeLists.txt: set line 13 to the "include" directory of libigl (this is a header only library)
5. "cd ${ROOT}/FitAdam/ && mkdir build && cd build"
6. "cmake .."
7. "make -j12"

# Usage
1. Suppose the video to be tested is named "${seqName}.mp4". Place it in "${ROOT}/${seqName}/${seqName}.mp4".
2. If the camera intrinsics is known, put it in "${ROOT}/${seqName}/calib.json" (refer to "POF/calib.json" for example); otherwise, a default camera intrinsics will be used.
3. In ${ROOT}, run "bash run_pipeline.sh ${seqName}"; if the subject in the video shows only upper body, run "bash run_pipeline.sh ${seqName} -f".

# Docker Image
1. Install NVIDIA Docker
2. Build the docker image
```
  docker build . --tag mtc
```
3. Running the docker image:
```
  xhost local:root
  docker run --gpus 0 -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all mtc
```
4. Once inside (should be in /opt/mtc by default):
```
  bash run_pipeline.sh example_speech -f
```
Tested on Ubuntu 16.04 and 18.04 with Titan Xp and Titan X Maxwell (External w/Razer Core). 

# Examples
"download.sh" automatically download 2 example videos to test. After successful installation run
```
bash run_pipeline.sh example_dance
```
or
```
bash run_pipeline.sh example_speech -f
```

# License and Citation
This code can only be used for **non-commercial research purposes**. If you use this code in your research, please cite the following papers.
```
@inproceedings{xiang2019monocular,
  title={Monocular total capture: Posing face, body, and hands in the wild},
  author={Xiang, Donglai and Joo, Hanbyul and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@inproceedings{joo2018total,
  title={Total capture: A 3d deformation model for tracking faces, hands, and bodies},
  author={Joo, Hanbyul and Simon, Tomas and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

Some part of this code is modified from [lmb-freiburg/hand3d](https://github.com/lmb-freiburg/hand3d).

# Adam Model
We use the deformable human model [**Adam**](http://www.cs.cmu.edu/~hanbyulj/totalcapture/) in this code.

**The relationship between Adam and SMPL:** The body part of Adam is derived from [SMPL](http://smpl.is.tue.mpg.de/license_body) model by Loper et al. 2015. It follows SMPL's body joint hierarchy, but uses a different joint regressor. Adam does not contain the original SMPL model's shape and pose blendshapes, but uses its own version trained from Panoptic Studio database.

**The relationship between Adam and FaceWarehouse:** The face part of Adam is derived from [FaceWarehouse](http://kunzhou.net/zjugaps/facewarehouse/). In particular, the mesh topology of face of Adam is a modified version of the learned model from FaceWarehouse dataset. Adam does not contain the blendshapes of the original FaceWarehouse data, and facial expression of Adam model is unavailable due to copyright issues.

The Adam model is shared for research purpose only, and cannot be used for commercial purpose. Redistributing the original or modified version of Adam is also not allowed without permissions. 

# Special Notice
1. In our code, the output of ceres::AngleAxisToRotationMatrix is always a RowMajor matrix, while the function is designed for a ColMajor matrix. To account for this, please treat our output pose parameters as the opposite value. In other words, before exporting our pose parameter to other softwares, please multiply them by -1.

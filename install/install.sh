# Install CuDNN v8.9.7
sudo apt-get install zlib1g
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Install Cuvid
sudo cp -r cuvid/lib/* /usr/local/cuda/lib64
sudo cp -r cuvid/include/* /usr/local/cuda/include

# Install OpenCV
## Install required tools and packages
sudo apt-get update
sudo apt install cmake
sudo apt install gcc g++
sudo apt install python3 python3-dev python3-numpy
sudo apt install libavcodec-dev libavformat-dev libswscale-dev
sudo apt install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
sudo apt install libgtk-3-dev
sudo apt install libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev

## Download OpenCV sources
sudo -s
cd /opt
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir release && cd release

## make install and link pkgconfig file
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D WITH_TBB=ON \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D NVCUVID_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D BUILD_opencv_java=OFF \
      -D BUILD_ZLIB=ON \
      -D BUILD_TIFF=ON \
      -D WITH_GTK=ON \
      -D WITH_NVCUVID=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_1394=ON \
      -D BUILD_PROTOBUF=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_PC_FILE_NAME=opencv4.pc \
      -D OPENCV_ENABLE_NONFREE=OFF \
      -D WITH_GSTREAMER=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_CUDNN=ON \
      -D BUILD_opencv_dnn=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      /opt/opencv/

make -j4
make install
ldconfig
exit
cd ~
ls /usr/local/lib/pkgconfig/
sudo cp /usr/local/lib/pkgconfig/opencv4.pc  /usr/lib/x86_64-linux-gnu/pkgconfig/opencv.pc
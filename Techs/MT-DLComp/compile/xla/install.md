# Things to run for installation

## Install
```shell
conda create -n tf2-cpu python=3.9
conda activate tf2-cpu
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.5.0-cp39-cp39-manylinux2010_x86_64.whl
pip install tqdm onnx onnxruntime
conda install -c anaconda bazel==3.7.2
```

## Install onnx-tf
```shell
git clone git@github.com:Wilbur-Django/onnx-tensorflow.git
cd onnx-tensorflow
pip install -e .
```

## Configure tensorflow
```shell
git clone --depth=1 git@github.com:tensorflow/tensorflow.git
cd tensorflow
./configure
```
Accept all the default settings when running ./configure

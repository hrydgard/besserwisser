BesserWisser
============

A simple implementation of deep neural networks. Made this in order to understand them better.

For production use, you're probably better of with something like Caffe or Keras.

Features working:

  * Load MNIST data set
	* Mini-batch training
	* Multi-layer (fully connected) neural networks
	* RELU and Sigmoid activation functions (RELU recommended)
	* SVM loss function
	* Inference
	* AVX acceleration of nearly all functionality

Future plans:

	* Vectorize mini-batch training properly
	* API and a workable command line/config file user interface
  * GPU training and inference using Vulkan
	* Multi-threaded CPU training
	* Convolutional nets

To run the example, first run get-mnist.sh to get the MNIST dataset (or download it manually). If you're on Windows, get-mnist.sh will run under WSL. Also make sure to run ```git submodule update --init --recursive``` before building.

To build and run on Windows:

  * With VS2017, use File->Open Folder to open as a CMake project.
  * Choose x64-Release and to the right of that make sure to choose the right output to run.
  * Ctrl+F5

To build on other platforms:

  * ```cmake .```
	* make

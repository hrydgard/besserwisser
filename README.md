SimpleNet
=========

A very simple implementation of neural networks. Made this in order to understand them better.

For production use, you're currently probably better of with something like Caffe or Keras.

Features working:

  * Load MNIST data set
	* Mini-batch training
	* Multi-layer (fully connected) neural networks
	* RELU and Sigmoid activation functions (RELU recommended)
	* SVM loss function (SoftMax not yet working)
	* Inference
	* AVX acceleration of nearly all functionality

Future plans:

	* Vectorize mini-batch training properly
	* Nicer API and command line user interface
  * GPU training and inference using Vulkan
	* Multi-threaded CPU training
	* Convolutional nets

To run the example, first run get-mnist.sh to get the MNIST dataset (or download it manually). If you're on Windows, get-mnist.sh will run under WSL.

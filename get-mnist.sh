# Use this script to easily download and unpack the MNIST dataset.

echo "Downloading MNIST training/test set..."
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --output train-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --output train-labels-idx1-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz --output t10k-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz --output t10k-labels-idx1-ubyte.gz

echo "Unpacking MNIST training/test sets..."
gunzip train-images-idx3-ubyte.gz
gunzip train-images-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-images-idx1-ubyte.gz

echo "Done."

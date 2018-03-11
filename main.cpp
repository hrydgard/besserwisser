// Neural net code by Henrik Rydgård

// A collection of useful links to understand this stuff:
// CS231n (free Stanford course material): http://cs231n.github.io/
// Debugging help: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
// 3blue1brown videos: 

#include <cmath>
#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <cstdio>

#include "math_util.h"
#include "mnist.h"

#include "layer.h"
#include "network.h"
#include "train.h"

int RunConvTest(int argc, const char *argv[]) {
	// TODO

	return 0;
}

// simple architectures:
// (note: No RELU before the loss function)
// INPUT -> FC -> LOSS  (89% on MNIST)
// INPUT -> FC -> RELU -> FC -> LOSS (98% on MNIST)
int RunMnist(int argc, const char *argv[]) {
	std::string mnist_root = "";
	if (argc > 1)
		mnist_root = std::string(argv[1]);
	if (mnist_root.empty())
		mnist_root = "C:/dev/MNIST";

	// http://yann.lecun.com/exdb/mnist/
	// The expected error rate for a pure linear classifier is 12% and we achieve that
	// with both fast back propagation and of course brute force.
	// The expected error rate for a 2-layer with 100 nodes is 2% which we do achieve
	// with the right hyperparameters!

	printf("Loading MNIST...\n");
	DataSet trainingSet;
	trainingSet.images = LoadMNISTImages(mnist_root + "/train-images.idx3-ubyte");
	trainingSet.labels = LoadMNISTLabels(mnist_root + "/train-labels.idx1-ubyte");
	assert(trainingSet.images.size() == trainingSet.images.size());

	DataSet testSet;
	testSet.images = LoadMNISTImages(mnist_root + "/t10k-images.idx3-ubyte");
	testSet.labels = LoadMNISTLabels(mnist_root + "/t10k-labels.idx1-ubyte");
	assert(testSet.images.size() == testSet.images.size());

	NeuralNetwork network;
	network.hyperParams.regStrength = 0.001f;
	network.hyperParams.miniBatchSize = 32;
	network.hyperParams.weightInitScale = 0.05f;

	ImageLayer imageLayer(&network);
	imageLayer.numInputs = 0;
	imageLayer.numData = 28 * 28 + 1;  // + 1 for bias trick
	network.layers.push_back(&imageLayer);

#if 1  // 2-layer network
	FcLayer hiddenLayer(&network);
	hiddenLayer.numInputs = 28 * 28 + 1;
	hiddenLayer.numData = 100;
	hiddenLayer.skipBackProp = true;
	network.layers.push_back(&hiddenLayer);

	LeakyReluLayer activation(&network);
	activation.numInputs = hiddenLayer.numData;
	activation.numData = hiddenLayer.numData;
	network.layers.push_back(&activation);

#if 0  // Extend to 3 layers. Does not seem to improve perf.
	FcLayer hiddenLayer2(&network);
	hiddenLayer2.numInputs = 100;
	hiddenLayer2.numData = 100;
	network.layers.push_back(&hiddenLayer2);

	ReluLayer activation2(&network);
	activation2.numInputs = hiddenLayer2.numData;
	activation2.numData = hiddenLayer2.numData;
	network.layers.push_back(&activation2);
#endif

	FcLayer linearLayer(&network);
	linearLayer.numInputs = hiddenLayer.numData;
	linearLayer.numData = 10;
	network.layers.push_back(&linearLayer);

	FcLayer *testLayer = (FcLayer *)&hiddenLayer;
#else
	FcLayer linearLayer(&network);
	linearLayer.numInputs = 28 * 28 + 1;
	linearLayer.numData = 10;
	network.layers.push_back(&linearLayer);

	FcLayer *testLayer = (FcLayer *)&linearLayer;
#endif

	SVMLossLayer lossLayer(&network);
	lossLayer.numInputs = 10;
	lossLayer.numData = 1;
	network.layers.push_back(&lossLayer);

	network.InitializeNetwork();

#if 1
	if (!RunBruteForceTest(network, testLayer, trainingSet)) {
		network.layers[0]->data = nullptr;  // Don't want to autodelete this..
		while (true);  // wait for Ctrl+C.
		return 0;
	}
#endif

	// TODO: Add support for separated dev and test sets if available (or generate them).
	TrainAndEvaluateNetworkStochastic(network, trainingSet, testSet);
	return 0;
}

int main(int argc, const char *argv[]) {
	printf("BesserWisser - a neural network implementation by Henrik Rydgård\n");
	RunMnist(argc, argv);
	while (true);  // wait for Ctrl+C.
	printf("Done.\n");
	return 0;
}

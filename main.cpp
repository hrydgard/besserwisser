// Neural net code by Henrik Rydgård

// A collection of useful links to understand this stuff:
// CS231n (free Stanford course material): http://cs231n.github.io/
// Debugging help: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

#include <cmath>
#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

#include "math_util.h"
#include "mnist.h"

#include "layer.h"
#include "network.h"
#include "train.h"

// simple architectures:
// (note: No RELU before the loss function)
// INPUT -> FC -> RELU -> FC -> RELU -> FC -> LOSS
// INPUT -> FC -> RELU -> FC -> LOSS
// INPUT -> FC -> LOSS

int main() {
	// http://yann.lecun.com/exdb/mnist/
	// The expected error rate for a pure linear classifier is 12% and we achieve that
	// with both fast back propagation and of course brute force.
	// The expected error rate for a 2-layer with 100 nodes is 2% which we do achieve
	// with the right hyperparameters!
	DataSet trainingSet;
	trainingSet.images = LoadMNISTImages("C:/dev/MNIST/train-images.idx3-ubyte");
	trainingSet.labels = LoadMNISTLabels("C:/dev/MNIST/train-labels.idx1-ubyte");
	assert(trainingSet.images.size() == trainingSet.images.size());

	DataSet testSet;
	testSet.images = LoadMNISTImages("C:/dev/MNIST/t10k-images.idx3-ubyte");
	testSet.labels = LoadMNISTLabels("C:/dev/MNIST/t10k-labels.idx1-ubyte");
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

	ReluLayer activation(&network);
	activation.numInputs = hiddenLayer.numData;
	activation.numData = hiddenLayer.numData;
	network.layers.push_back(&activation);

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

	static const char *labelNames[10] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

	int subsetSize = network.hyperParams.miniBatchSize;

	std::vector<std::vector<int>> subsets = GenerateRandomSubsets(trainingSet.images.size(), subsetSize);

	RunStats stats;

	MiniBatch testSubset;
	testSubset.dataSet = &testSet;
	testSubset.indices = GetFullSet(testSubset.dataSet->images.size());

	stats = {};
	float lossOnTestset = ComputeLossOverMinibatch(network, testSubset, &stats);
	printf("Loss on test set before training: %f\n", lossOnTestset);
	stats.Print();

	MiniBatch subset;
	subset.dataSet = &trainingSet;
	subset.indices = { 1, 2 };

	// Run the network first forward then backwards, then compute the brute force gradient and compare.
	printf("Fast gradient (b)...\n");
	network.ClearDeltaWeightSum();
	for (auto index : subset.indices) {
		network.layers[0]->data = subset.dataSet->images[index].data;
		network.layers.back()->label = subset.dataSet->labels[index];
		network.RunForwardPass();
		network.RunBackwardPass();  // Accumulates delta weights.
	}
	network.ScaleDeltaWeightSum(1.0f / subset.indices.size());

	float *deltaWeightSum = new float[testLayer->numWeights]{};
	printf("Computing test gradient over %d examples by brute force (a)...\n", (int)subset.indices.size());
	ComputeDeltaWeightSumBruteForce(network, subset, testLayer, deltaWeightSum);
	int diffCount = DiffVectors(deltaWeightSum, testLayer->deltaWeightSum, testLayer->numWeights, 0.01f, 200);
	printf("Done with test.\n");

	if (diffCount > 1000) {
		network.layers[0]->data = nullptr;  // Don't want to autodelete this..
		while (true);
		return 0;
	}

	delete[] deltaWeightSum;

	float trainingSpeed = 0.015f;

	int rounds = (int)subsets.size();
	for (int epoch = 0; epoch < 100; epoch++) {
		// Decay training speed every 10 epochs. TODO: Make tunable.
		if (epoch != 0 && (epoch % 10 == 0))
			trainingSpeed *= 0.75f;

		printf("Epoch %d, trainingSpeed=%f\n", epoch + 1, trainingSpeed);
		for (int i = 0; i < rounds; i++) {
			int subsetIndex = i % subsets.size();
			// printf("Round %d/%d (subset %d/%d)\n", i + 1, rounds, subsetIndex + 1, (int)subsets.size());
			// Train on different subsets each round (stochastic gradient descent)
			subset.indices = subsets[subsetIndex];
			//float loss = ComputeLossOverSubset(network, subset);

			// TrainLayerBruteForce(network, subset, &linearLayer, trainingSpeed);
			TrainNetworkOnMinibatch(network, subset, trainingSpeed);
			// UpdateLayerFast(network, subset, &linearLayer, trainingSpeed);
			/*
			stats = {};
			float lossAfterTraining = ComputeLossOverSubset(network, subset, &stats);

			PrintFloatVector("Neurons", network.layers.back()->data, network.layers.back()->numData);
			printf("Loss before: %0.3f\n", loss);
			printf("Loss after: %0.3f\n", lossAfterTraining);
			stats.Print();*/
		}
		subsets = GenerateRandomSubsets(trainingSet.images.size(), subsetSize);
		printf("Running on testset (%d images)...\n", (int)testSubset.dataSet->images.size());
		stats = {};
		lossOnTestset = ComputeLossOverMinibatch(network, testSubset, &stats);
		printf("Loss on testset: %f\n", lossOnTestset);
		stats.Print();
		PrintFloatVector("hidden", hiddenLayer.data, hiddenLayer.numData);
	}

	printf("Done.");
	
	while (true);
	/*
	// Evaluate network on the test set.
	float totalLoss = 0.0f;
	for (int i = 0; i < testImages.size(); i++) {
		assert(imageLayer.numNeurons == testImages[i].size);
		imageLayer.neurons = testImages[i].data;
		RunNetwork(network);
		// int inferredLabel = Judge(fcLayer.neurons, 10);
		float loss = network.lossFunction(fcLayer.neurons, 10, testLabels[i]);
		totalLoss += loss;
	}
	totalLoss /= trainLabels.size();
	*/
	return 0;
}

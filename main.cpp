﻿#include <cmath>
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

struct RunStats {
	int correct;
	int wrong;
	int correctCount[10];
	void Print() {
		printf("Results: %d correct, %d wrong\n", correct, wrong);
		for (int i = 0; i < ARRAY_SIZE(correctCount); i++) {
			// printf("%d: %d\n", i, correctCount[i]);
		}
	}
};

// Runs the forward pass.
float ComputeDataLoss(NeuralNetwork &network, const DataSet &dataSet, int index, RunStats *stats = nullptr) {
	assert(network.layers[0]->type == LayerType::IMAGE);
	Layer *scoreLayer = network.layers[network.layers.size() - 2];
	Layer *finalLayer = network.layers.back();
	// Last layer must be a loss layer.
	assert(finalLayer->type == LayerType::SVM_LOSS || finalLayer->type == LayerType::SOFTMAX_LOSS);
	network.layers[0]->data = dataSet.images[index].data;
	finalLayer->label = dataSet.labels[index];
	network.RunForwardPass();
	return finalLayer->data[0];
}

float ComputeRegularizationLoss(NeuralNetwork &network) {
	// Penalize with regularization term 0.5lambdaX^2 to discourage high volume noise in the matrix.
	// Note that its gradient will be simply lambdaX.
	// TODO: AVX!
	double regSum = 0.0;
	for (size_t i = 0; i < network.layers.size(); i++) {
		Layer &layer = *network.layers[i];
		regSum += layer.GetRegularizationLoss();
	}
	return 0.5f * network.hyperParams.regStrength * (float)regSum;
}

float ComputeLossOverSubset(NeuralNetwork &network, const Subset &subset, RunStats *stats = nullptr) {
	Layer *scoreLayer = network.layers[network.layers.size() - 2];
	Layer *finalLayer = network.layers.back();
	// Last layer must be a loss layer.
	assert(finalLayer->type == LayerType::SVM_LOSS || finalLayer->type == LayerType::SOFTMAX_LOSS);

	// Computes the total loss as a single number over a set of input images.
	// Should probably do it as a vector instead, it's a bit crazy that this works as is.
	float totalLoss = 0.0f;
	for (int i = 0; i < subset.indices.size(); i++) {
		int index = subset.indices[i];
		float loss = ComputeDataLoss(network, *subset.dataSet, index, stats);
		if (stats) {
			int label = FindMaxIndex(scoreLayer->data, scoreLayer->numData);
			assert(label >= 0);
			if (label == subset.dataSet->labels[index]) {
				stats->correct++;
				stats->correctCount[label]++;
			} else {
				stats->wrong++;
			}
		}
		totalLoss += loss;
	}
	totalLoss /= subset.indices.size();

	totalLoss += ComputeRegularizationLoss(network);
	return totalLoss;
}

// TODO: Change this to compare directly to the most recently computed gradient
// Computes the sum of gradients from a minibatch.
void ComputeDeltaWeightSumBruteForce(NeuralNetwork &network, const Subset &subset, Layer *layer, float *gradient) {
	assert(layer->type == LayerType::FC);
	FcLayer *fcLayer = dynamic_cast<FcLayer *>(layer);

	const float diff = 0.001f;
	const float inv2Diff = 1.0f / (2.0f * diff);
	size_t size = fcLayer->numWeights;
	for (int i = 0; i < size; i++) {
		float origWeight = fcLayer->weights[i];
		// Tweak up and compute loss
		fcLayer->weights[i] = origWeight + diff;
		float up = ComputeLossOverSubset(network, subset);
		// Tweak down and compute loss
		fcLayer->weights[i] = origWeight - diff;
		float down = ComputeLossOverSubset(network, subset);
		// Restore and compute gradient.
		fcLayer->weights[i] = origWeight;
		gradient[i] = (up - down) * inv2Diff;
	}
	PrintFloatVector("Weights", fcLayer->weights, size, 10);
	PrintFloatVector("Gradient", gradient, size, 10);
}

// Train a single layer using a minibatch.

void TrainNetworkFast(NeuralNetwork &network, const Subset &subset, float speed) {
	network.ClearDeltaWeightSum();
	for (auto index : subset.indices) {
		network.layers[0]->data = subset.dataSet->images[index].data;
		network.layers.back()->label = subset.dataSet->labels[index];
		network.RunForwardPass();
		network.RunBackwardPass();  // Accumulates delta weights
	}
	network.ScaleDeltaWeightSum(1.0f / subset.indices.size());

	// Update all training weights.
	for (auto *layer : network.layers) {
		layer->UpdateWeights(speed);
	}
}

void TrainLayerBruteForce(NeuralNetwork &network, const Subset &subset, Layer *layer, float speed) {
	// Hacky
	FcLayer *fcLayer = (FcLayer *)layer;

	size_t size = fcLayer->numWeights;
	float *gradient = new float[size];
	ComputeDeltaWeightSumBruteForce(network, subset, layer, gradient);
	// Simple gradient descent.
	// Saxpy(size, -speed, gradient, layer->weights);
	for (int i = 0; i < size; i++) {
		fcLayer->weights[i] -= gradient[i] * speed;
	}
	delete[] gradient;
}

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
	/*
	network.hyperParams.miniBatchSize = 16;
	network.hyperParams.weightInitScale = 0.1f;
	*/
	ImageLayer imageLayer(&network);
	imageLayer.numInputs = 0;
	imageLayer.numData = 28 * 28 + 1;  // + 1 for bias trick
	network.layers.push_back(&imageLayer);

#if 1
	FcLayer hiddenLayer(&network);
	hiddenLayer.numInputs = 28 * 28 + 1;
	hiddenLayer.numData = 100;
	hiddenLayer.skipBackProp = true;
	network.layers.push_back(&hiddenLayer);

	SigmoidLayer activation(&network);
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

	Subset testSubset;
	testSubset.dataSet = &testSet;
	testSubset.indices = GetFullSet(testSubset.dataSet->images.size());

	stats = {};
	float lossOnTestset = ComputeLossOverSubset(network, testSubset, &stats);
	printf("Loss on test set before training: %f\n", lossOnTestset);
	stats.Print();

	Subset subset;
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
			TrainNetworkFast(network, subset, trainingSpeed);
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
		lossOnTestset = ComputeLossOverSubset(network, testSubset, &stats);
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

#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

// Training and evaluation utilities.
#include "network.h"

struct DataSet {
	std::vector<Blob> images;
	std::vector<uint8_t> labels;
};

// Defines a subset of a dataset.
struct MiniBatch {
	const DataSet *dataSet;
	std::vector<int> indices;
};

struct RunStats {
	int correct;
	int wrong;
	void Print() {
		printf("Results: %d correct, %d wrong\n", correct, wrong);
	}
};

// Does what it says on the tin. Does not run a backwards pass. After this,
// you can read out the result vector from data in the next-to-last layer (assuming a normal network).
float ComputeLossOverMinibatch(NeuralNetwork &network, const MiniBatch &subset, RunStats *stats = nullptr);

// A simple way to check that back propagation matches brute force.
bool RunBruteForceTest(NeuralNetwork &network, FcLayer *testLayer, const DataSet &dataSet);

// This is the normal way to train a classifier network in minibatches.
void TrainAndEvaluateNetworkStochastic(NeuralNetwork &network, const DataSet &trainingSet, const DataSet &testSet, int maxEpochs = 100);

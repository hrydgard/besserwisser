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
	~MiniBatch() {
		delete[] blobs;
		delete[] labels;
	}
	const DataSet *dataSet;
	std::vector<int> indices;

	void Extract() {
		blobs = new const Blob*[indices.size()];
		labels = new int[indices.size()];
		for (size_t i = 0; i < indices.size(); i++) {
			blobs[i] = &dataSet->images[indices[i]];
			labels[i] = dataSet->labels[indices[i]];
		}
	}

	const Blob **blobs = nullptr;
	int *labels = nullptr;
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
void TrainAndEvaluateNetworkStochastic(NeuralNetwork &network, const DataSet &dataSet, const DataSet &testSet, int maxEpochs, int miniBatchSize);

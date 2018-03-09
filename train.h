#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

// Training and evaluation utilities.
#include "network.h"

struct DataSet {
	std::vector<DataVector> images;
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

// Does what it says on the tin. Does not run a backwards pass.
float ComputeLossOverMinibatch(NeuralNetwork &network, const MiniBatch &subset, RunStats *stats = nullptr);

// Utility used for validation of the real training code. Will simply do a brute force gradient calculation
// by disturbing and resetting every training weight of the chosen layer (finite difference method).
void ComputeDeltaWeightSumBruteForce(NeuralNetwork &network, const MiniBatch &subset, Layer *layer, float *gradient);

// Trains all the weights in a network by running both a forward and a backward pass for every item
// in the minibatch. Hyperparameters for training are configured directly on the network object.
void TrainNetworkOnMinibatch(NeuralNetwork &network, const MiniBatch &subset, float speed);

// Trains a single layer of a network using ComputeDeltaWeightSumBruteForce. This does work just fine
// but is incredibly slow. Not really useful once you've validated all your back propagation.
void TrainLayerBruteForce(NeuralNetwork &network, const MiniBatch &subset, Layer *layer, float speed);

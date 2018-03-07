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

struct DataSet {
	std::vector<DataVector> images;
	std::vector<uint8_t> labels;
};

struct Subset {
	DataSet *dataSet;
	std::vector<int> indices;
};

struct RunStats {
	int correct;
	int wrong;
	int correctCount[10];
	void Print() {
		printf("Results: %d correct, %d wrong\n", correct, wrong);
		for (int i = 0; i < ARRAY_SIZE(correctCount); i++) {
			printf("%d: %d\n", i, correctCount[i]);
		}
	}
};

float ComputeDataLoss(NeuralNetwork &network, const DataSet &dataSet, int index, RunStats *stats = nullptr) {
	assert(network.layers[0]->type == LayerType::IMAGE);
	Layer *scoreLayer = network.layers[network.layers.size() - 2];
	Layer *finalLayer = network.layers.back();
	// Last layer must be a loss layer.
	assert(finalLayer->type == LayerType::SVM_LOSS || finalLayer->type == LayerType::SOFTMAX_LOSS);
	network.layers[0]->neurons = dataSet.images[index].data;
	finalLayer->label = dataSet.labels[index];
	network.RunForwardPass();
	float loss = Sum(finalLayer->neurons, finalLayer->numNeurons);
	return loss;
}

float ComputeRegularizationLoss(NeuralNetwork &network) {
	// Penalize with regularization term 0.5lambdaX^2 to discourage high volume noise in the matrix.
	// Note that its gradient will be simply lambdaX.
	// TODO: AVX!
	double regSum = 0.0;
	for (size_t i = 0; i < network.layers.size(); i++) {
		Layer &layer = *network.layers[i];
		if (layer.type == LayerType::FC) {
			for (size_t j = 0; j < layer.numWeights; j++) {
				regSum += 0.5f * network.hyperParams.regStrength * sqr(layer.weights[j]);
			}
		}
	}
	return (float)regSum;
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
			int label = FindMaxIndex(scoreLayer->neurons, scoreLayer->numNeurons);
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

void ComputeGradientBruteForce(NeuralNetwork &network, const Subset &subset, Layer *layer, float *gradient) {
	assert(layer->type == LayerType::FC);

	const float diff = 0.001f;
	const float inv2Diff = 1.0f / (2.0f * diff);
	size_t size = layer->numWeights;
	for (int i = 0; i < size; i++) {
		float origWeight = layer->weights[i];
		// Tweak up and compute loss
		layer->weights[i] = origWeight + diff;
		float up = ComputeLossOverSubset(network, subset);
		// Tweak down and compute loss
		layer->weights[i] = origWeight - diff;
		float down = ComputeLossOverSubset(network, subset);
		// Restore and compute gradient.
		layer->weights[i] = origWeight;
		gradient[i] = (up - down) * inv2Diff;
	}
	PrintFloatVector("Weights", layer->weights, size, 10);
	PrintFloatVector("Gradient", gradient, size, 10);
}

void UpdateLayerFast(NeuralNetwork &network, const Subset &subset, Layer *layer, float speed) {
	size_t size = layer->numWeights;
	// Simple gradient descent.
	// Saxpy(size, -speed, gradient, layer->weights);
	for (int i = 0; i < size; i++) {
		layer->weights[i] -= layer->gradient[i] * speed;
	}
}

void UpdateLayerBruteForce(NeuralNetwork &network, const Subset &subset, Layer *layer, float speed) {
	size_t size = layer->numWeights;
	float *gradient = new float[size];
	ComputeGradientBruteForce(network, subset, layer, gradient);
	// Simple gradient descent.
	// Saxpy(size, -speed, gradient, layer->weights);
	for (int i = 0; i < size; i++) {
		layer->weights[i] -= gradient[i] * speed;
	}
	delete[] gradient;
}

// simple architectures:
// (note: No RELU before the loss function)
// INPUT -> FC -> RELU -> FC -> RELU -> FC -> LOSS
// INPUT -> FC -> RELU -> FC -> LOSS
// INPUT -> FC -> LOSS

int main() {
	DataSet trainingSet;
	trainingSet.images = LoadMNISTImages("C:/dev/MNIST/train-images.idx3-ubyte");
	trainingSet.labels = LoadMNISTLabels("C:/dev/MNIST/train-labels.idx1-ubyte");
	assert(trainingSet.images.size() == trainingSet.images.size());

	DataSet testSet;
	testSet.images = LoadMNISTImages("C:/dev/MNIST/t10k-images.idx3-ubyte");
	testSet.labels = LoadMNISTLabels("C:/dev/MNIST/t10k-labels.idx1-ubyte");
	assert(testSet.images.size() == testSet.images.size());

	NeuralNetwork network;
	ImageLayer imageLayer(&network);
	imageLayer.inputDim = ivec3{ 28, 28, 1 };
	imageLayer.numInputs = 0;
	imageLayer.numNeurons = 28 * 28 + 1;  // + 1 for bias trick
	network.layers.push_back(&imageLayer);

	/*
	NeuralLayer hiddenLayer{ LayerType::FC, ivec3{100,1,1} };
	hiddenLayer.numInputs = 28 * 28;
	hiddenLayer.numNeurons = 100;
	network.layers.push_back(&hiddenLayer);

	NeuralLayer relu{ LayerType::RELU };
	relu.numInputs = 100;
	relu.numNeurons = 100;
	network.layers.push_back(&relu);

	NeuralLayer fcLayer{ LayerType::FC, ivec3{ 32,32,1 } };
	fcLayer.numInputs = 100;
	fcLayer.numNeurons = 10;
	network.layers.push_back(&fcLayer);
	*/
	FcLayer linearLayer(&network);
	linearLayer.numInputs = 28 * 28 + 1;
	linearLayer.numNeurons = 10;
	network.layers.push_back(&linearLayer);

	SVMLossLayer lossLayer(&network);
	lossLayer.numInputs = 10;
	lossLayer.numNeurons = 10;
	network.layers.push_back(&lossLayer);

	network.InitializeNetwork();

	static const char *labelNames[10] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

	int subsetSize = 32;

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
	subset.indices = { 1 };

	// Run the network first forward then backwards, then compute the brute force gradient and compare.
	printf("Fast gradient (b)...\n");
	network.ClearGradients();
	ComputeDataLoss(network, trainingSet, 1, &stats);
	network.RunBackwardPass();

	float *gradient = new float[linearLayer.numGradients]{};
	printf("Computing test gradient over %d examples by brute force (a)...\n", (int)subset.indices.size());
	ComputeGradientBruteForce(network, subset, &linearLayer, gradient);
	DiffVectors(gradient, linearLayer.gradient, linearLayer.numGradients, 0.01f, 2000);
	printf("Done with test.\n");
	delete[] gradient;

	float trainingSpeed = 0.05f;

	int rounds = 40;
	for (int i = 0; i < rounds; i++) {
		printf("Round %d/%d\n", i + 1, rounds);
		// Train on different subsets each round (stochastic gradient descent)
		subset.indices = subsets[i % subsets.size()];
		float loss = ComputeLossOverSubset(network, subset);

		UpdateLayerBruteForce(network, subset, &linearLayer, trainingSpeed);
		// UpdateLayerFast(network, subset, &linearLayer, trainingSpeed);

		stats = {};
		float lossAfterTraining = ComputeLossOverSubset(network, subset, &stats);

		PrintFloatVector("Neurons", network.layers.back()->neurons, network.layers.back()->numNeurons);
		printf("Loss before: %0.3f\n", loss);
		printf("Loss after: %0.3f\n", lossAfterTraining);
		stats.Print();
	}

	printf("Done.");
	
	printf("Running on testset (%d images)...", (int)testSubset.dataSet->images.size());
	stats = {};
	lossOnTestset = ComputeLossOverSubset(network, testSubset, &stats);
	printf("Loss on testset: %f", lossOnTestset);
	stats.Print();

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

#include <cmath>
#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

#include "math_util.h"
#include "mnist.h"

enum class LayerType {
	// These should be enough to get good results.
	UNDEFINED,
	IMAGE,  // Doesn't do anything, just a static image.
	FC,  // Fully connected. Same as a regular linear classifier matrix.
	RELU,  // Best non-linearity.
	SOFTMAX_LOSS,
	SVM_LOSS,  // This seems to converge faster than softmax.

	// Future
	CONV,
	MAXPOOL,  // Downsamples by 2x in X and Y but not Z.
};

class NeuralLayer {
public:
	virtual ~NeuralLayer() {
		delete[] neurons;
		delete[] weights;
		delete[] gradient;
	}

	virtual void Initialize() {}
	virtual void Forward(const float *input) = 0;   // input = The neurons from the previous layer
	virtual void Backward(const float *input);  // input = The gradients from the next layer

	LayerType type;
	ivec3 inputDim{};  // How the input should be interpreted. Important for some layer types.

	int numInputs = 0;
	int numNeurons = 0;
	int numWeights = 0;  // for convenience.
	int numGradients = 0;

	// State
	float *neurons = nullptr;  // Vector.

	// Trained. Only read from in forward pass, updated after computing gradients.
	float *weights = nullptr;  // Matrix (inputs * neurons)

	// Backward gradient
	float *gradient = nullptr;

	// Truth. Used by softmaxloss and svmloss layers.
	int label = -1;
};

class ReluLayer : public NeuralLayer {
public:
	ReluLayer() { type = LayerType::RELU; }
	void Forward(const float *input) override {
		// TODO: This can be very easily SIMD'd.
		for (int i = 0; i < numNeurons; i++) {
			neurons[i] = std::max(0.0f, input[i]);
		}
	}
	void Backward(const float *input) override {
		for (int i = 0; i < numNeurons; i++) {
			gradient[i] = neurons[i] > 0.0f ? 1.0f : 0.0f;
		}
	}
};

class ImageLayer : public NeuralLayer {
public:
	ImageLayer() { type = LayerType::IMAGE; }
	void Forward(const float *input) override {}
};

class FcLayer : public NeuralLayer {
public:
	FcLayer() { type = LayerType::FC; }
	void Forward(const float *input) override;
	void Backward(const float *input) override;
};

void FcLayer::Forward(const float *input) {
	// Just a matrix*vector multiplication.
	for (int y = 0; y < numNeurons; y++) {
		neurons[y] = DotAVX(input, &weights[y * numInputs], numInputs);
	}
}

void FcLayer::Backward(const float *input) {

}

class SVMLossLayer : public NeuralLayer {
public:
	SVMLossLayer() { type = LayerType::SVM_LOSS; }
	void Forward(const float *input) override;
	void Backward(const float *input) override;
};

void SVMLossLayer::Forward(const float *input) {
	float truth = input[label];
	assert(label != -1);
	for (size_t i = 0; i < numNeurons; i++) {
		if (i == label) {
			neurons[i] = 0.0f;
		} else {
			neurons[i] = std::max(0.0f, input[i] - truth + 1.0f);
		}
	}
}

void SVMLossLayer::Backward(const float *input) {

}

struct NeuralNetwork {
	std::vector<NeuralLayer *> layers;
};

void NeuralLayer::Backward(const float *gradients) {
	switch (type) {
	case LayerType::RELU:
		break;
	case LayerType::FC:
		break;
	case LayerType::IMAGE:
		// Nothing to do, images are read-only.
		break;
	default:
		// Do nothing.
		break;
	}
}

// Inference.
void RunForwardPass(NeuralNetwork &network) {
	for (int i = 1; i < network.layers.size(); i++) {
		network.layers[i]->Forward(network.layers[i - 1]->neurons);
	}
}

void RunBackwardPass(NeuralNetwork &network) {
	for (int i = network.layers.size() - 1; i >= 0; i++) {
		network.layers[i]->Backward((i < network.layers.size() - 1) ? network.layers[i + 1]->gradient : nullptr);
	}
}

void InitializeNetwork(NeuralNetwork &network) {
	for (int i = 1; i < network.layers.size(); i++) {
		NeuralLayer &layer = *network.layers[i];
		assert(layer.numInputs == network.layers[i - 1]->numNeurons);
		layer.neurons = new float[layer.numNeurons];
		switch (layer.type) {
		case LayerType::FC:
			layer.numWeights = layer.numNeurons * layer.numInputs;
			layer.weights = new float[layer.numWeights]{};
			layer.gradient = new float[layer.numWeights]{};  // Here we'll accumulate gradients before we do the adjust.
			GaussianNoise(layer.weights, layer.numWeights, 0.05f);
			break;
		case LayerType::RELU:
			assert(layer.numNeurons == layer.numInputs);
			layer.gradient = new float[layer.numNeurons]{};
			break;
		case LayerType::IMAGE:
			break;
		case LayerType::SOFTMAX_LOSS:
		case LayerType::SVM_LOSS:
			assert(layer.numNeurons == layer.numInputs);
			layer.gradient = new float[layer.numNeurons]{};
			break;
		}
	}
}

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

float ComputeLoss(NeuralNetwork &network, const Subset &subset, RunStats *stats = nullptr) {
	assert(network.layers[0]->type == LayerType::IMAGE);

	float regStrength = 0.5f;

	NeuralLayer *scoreLayer = network.layers[network.layers.size() - 2];
	NeuralLayer *finalLayer = network.layers.back();
	// Last layer must be a loss layer.
	assert(finalLayer->type == LayerType::SVM_LOSS || finalLayer->type == LayerType::SOFTMAX_LOSS);

	// Computes the total loss as a single number over a set of input images.
	// Should probably do it as a vector instead, it's a bit crazy that this works as is.
	float totalLoss = 0.0f;
	auto &images = subset.dataSet->images;
	auto &labels = subset.dataSet->labels;
	for (int i = 0; i < subset.indices.size(); i++) {
		int index = subset.indices[i];
		network.layers[0]->neurons = images[index].data;
		finalLayer->label = labels[index];
		RunForwardPass(network);
		totalLoss += Sum(finalLayer->neurons, finalLayer->numNeurons);
		if (stats) {
			int label = FindMaxIndex(scoreLayer->neurons, scoreLayer->numNeurons);
			assert(label >= 0);
			if (label == labels[index]) {
				stats->correct++;
				stats->correctCount[label]++;
			} else {
				stats->wrong++;
			}
		}

		// Penalize with regularization term 0.5lambdaX^2 to discourage high volume noise in the matrix.
		// Note that its gradient will be simply lambdaX.
		double regSum = 0.0;
		for (size_t i = 0; i < network.layers.size(); i++) {
			NeuralLayer &layer = *network.layers[i];
			for (size_t j = 0; j < layer.numWeights; j++) {
				regSum += 0.5f * regStrength * sqr(layer.weights[j]);
			}
		}
		totalLoss += (float)regSum;
	}
	totalLoss /= subset.indices.size();
	return totalLoss;
}

void ComputeGradientBruteForce(NeuralNetwork &network, const Subset &subset, int layerIndex, float *gradient) {
	const float diff = 0.0001f;
	const float inv2Diff = 1.0f / (2.0 * diff);
	NeuralLayer &layer = *network.layers[layerIndex];
	size_t size = layer.numWeights;
	for (int i = 0; i < size; i++) {
		float origWeight = layer.weights[i];
		// Tweak up and compute loss
		layer.weights[i] = origWeight + diff;
		float up = ComputeLoss(network, subset);
		// Tweak down and compute loss
		layer.weights[i] = origWeight - diff;
		float down = ComputeLoss(network, subset);
		// Restore and compute gradient.
		layer.weights[i] = origWeight;
		gradient[i] = (up - down) * inv2Diff;
	}
	PrintFloatVector("Weights", layer.weights, size, 10);
	PrintFloatVector("Gradient", gradient + size / 2 - 10, size/2, 10);
}

void UpdateLayerBruteForce(NeuralNetwork &network, const Subset &subset, int layerIndex, float speed) {
	NeuralLayer &layer = *network.layers[layerIndex];
	size_t size = layer.numWeights;
	float *gradient = new float[size];
	ComputeGradientBruteForce(network, subset, layerIndex, gradient);
	// Simple gradient descent.
	// Can be expressed as an axpy
	for (int i = 0; i < size; i++) {
		layer.weights[i] -= gradient[i] * speed;
	}
}

// simple architectures:

// INPUT -> FC -> RELU -> FC -> RELU -> FC
// INPUT -> FC -> RELU -> FC
// INPUT -> FC

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
	ImageLayer imageLayer;
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
	FcLayer linearLayer;
	linearLayer.numInputs = 28 * 28 + 1;
	linearLayer.numNeurons = 10;
	network.layers.push_back(&linearLayer);

	SVMLossLayer lossLayer{ };
	lossLayer.numInputs = 10;
	lossLayer.numNeurons = 10;
	network.layers.push_back(&lossLayer);

	InitializeNetwork(network);

	static const char *labelNames[10] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

	int subsetSize = 32;
	
	std::vector<std::vector<int>> subsets = GenerateRandomSubsets(trainingSet.images.size(), subsetSize);

	RunStats stats;

	Subset testSubset;
	testSubset.dataSet = &testSet;
	testSubset.indices = GetFullSet(testSubset.dataSet->images.size());

	stats = {};
	float lossOnTestset = ComputeLoss(network, testSubset, &stats);
	printf("Loss on test set before training: %f\n", lossOnTestset);
	stats.Print();

	Subset subset;
	subset.dataSet = &trainingSet;
	subset.indices = subsets[0];

	float trainingSpeed = 0.05f;

	int rounds = 40;
	for (int i = 0; i < rounds; i++) {
		printf("Round %d/%d\n", i + 1, rounds);
		// Train on different subsets each round (stochastic gradient descent)
		subset.indices = subsets[i % subsets.size()];
		float loss = ComputeLoss(network, subset);

		UpdateLayerBruteForce(network, subset, 1, trainingSpeed);

		stats = {};
		float lossAfterTraining = ComputeLoss(network, subset, &stats);
		PrintFloatVector("Neurons", network.layers.back()->neurons, network.layers.back()->numNeurons);
		printf("Loss before: %0.3f\n", loss);
		printf("Loss after: %0.3f\n", lossAfterTraining);
		stats.Print();
	}

	printf("Done.");
	
	printf("Running on testset (%d images)...", (int)testSubset.dataSet->images.size());
	stats = {};
	lossOnTestset = ComputeLoss(network, testSubset, &stats);
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

#include "train.h"
#include "layer.h"
#include "network.h"

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

float ComputeLossOverMinibatch(NeuralNetwork &network, const MiniBatch &subset, RunStats *stats) {
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
void ComputeDeltaWeightSumBruteForce(NeuralNetwork &network, const MiniBatch &subset, Layer *layer, float *gradient) {
	assert(layer->type == LayerType::FC);
	FcLayer *fcLayer = dynamic_cast<FcLayer *>(layer);

	const float diff = 0.001f;
	const float inv2Diff = 1.0f / (2.0f * diff);
	size_t size = fcLayer->numWeights;
	for (int i = 0; i < size; i++) {
		float origWeight = fcLayer->weights[i];
		// Tweak up and compute loss
		fcLayer->weights[i] = origWeight + diff;
		float up = ComputeLossOverMinibatch(network, subset);
		// Tweak down and compute loss
		fcLayer->weights[i] = origWeight - diff;
		float down = ComputeLossOverMinibatch(network, subset);
		// Restore and compute gradient.
		fcLayer->weights[i] = origWeight;
		gradient[i] = (up - down) * inv2Diff;
	}
	PrintFloatVector("Weights", fcLayer->weights, size, 10);
	PrintFloatVector("Gradient", gradient, size, 10);
}

// Train a single layer using a minibatch.

void TrainNetworkOnMinibatch(NeuralNetwork &network, const MiniBatch &subset, float speed) {
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

void TrainLayerBruteForce(NeuralNetwork &network, const MiniBatch &subset, Layer *layer, float speed) {
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
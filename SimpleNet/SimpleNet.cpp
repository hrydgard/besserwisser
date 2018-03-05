// SimpleNet.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

struct ivec3 {
	int x, y, z;
};

struct DataVector {
	~DataVector() {
		delete[] data;
	}
	// Probably quite useless.
	void SetToClassification(int x, size_t count) {
		if (size != count) {
			delete[] data;
			size = count;
			data = new float[count] {};
		} else {
			std::fill(data, data + count, 0.0f);
		}
		data[x] = 1.0f;
	}
	float *data = nullptr;
	size_t size = 0;
	ivec3 dim{};  // Dimensions the data should be interpreted at. Will tag along on the ride.
};

inline float RELU(float x) {
	return x > 0.0f ? x : 0.0f;
}

inline float ByteToFloat(uint8_t b) {
	return (float)b * (1.0f / 255.0f);
}

// TODO: SSE2 it up.
inline float Dot(const float *a, const float *b, size_t size) {
	float sum = 0.0f;
	for (size_t i = 0; i < size; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

inline uint32_t swap32(uint32_t x) {
	return _byteswap_ulong(x);
}

inline uint32_t readBE32(FILE *f) {
	uint32_t x;
	fread(&x, 1, 4, f);
	return swap32(x);
}

struct Tensor {
	Tensor() : w(0), h(0), d(0), data(nullptr) {}
	Tensor(int _w, int _h, int _d) : w(_w), h(_h), d(_d) {
		data = new float[w * h * d];
	}
	~Tensor() {
		delete[] data;
	}

	void operator=(Tensor &&tensor) {
		delete[] data;
		data = tensor.data;
		w = tensor.w;
		h = tensor.h;
		d = tensor.d;
	}
	Tensor(Tensor&& tensor) noexcept : data(tensor.data), w(tensor.w), h(tensor.h), d(tensor.d) {}

	int GetDataSize() const {
		return w * h * d;
	}

	int w, h, d;
	float *data;

	float &At(int x, int y, int z) {
		return data[z * w * h + y * w + x];
	}
};

enum class LayerType {
	UNDEFINED,
	FC,  // Fully connected.
	CONV,
	POOL,  // Downsamples by 2x in X and Y but not Z
	RELU,
	IMAGE,  // Doesn't do anything, just a static image.
};

typedef float(*LossFunction)(const float *data, size_t size, int correctLabel);

struct NeuralLayer {
	LayerType type = LayerType::UNDEFINED;
	ivec3 inputDim{};  // How the input should be interpreted. Important for some layer types.

	int numInputs = 0;
	int numNeurons = 0;

	// Trained:
	float *weights = nullptr;  // Matrix (inputs * neurons)
	float *biases = nullptr;   // Vector. Same size as neurons.

	float *neurons = nullptr;  // Vector.
};

struct NeuralNetwork {
	std::vector<NeuralLayer *> layers;
	LossFunction lossFunction;
};

struct TrainingSet {
	
};

// Layers own their own outputs.
void ApplyLayer(const NeuralLayer &layer, const float *input) {
	Tensor output;
	switch (layer.type) {
	case LayerType::RELU: {
		size_t sz = output.GetDataSize();
		for (int i = 0; i < layer.numInputs; i++) {
			layer.neurons[i] = RELU(input[i]);
		}
		break;
	}
	case LayerType::FC: {
		// Essentially a matrix multiplication and a vector addition.
		for (int y = 0; y < layer.numNeurons; y++) {
			layer.neurons[y] = Dot(input, &layer.weights[y * layer.numInputs], layer.numInputs) + layer.biases[y];
		}
		break;
	}
	case LayerType::CONV:
		break;
	case LayerType::POOL:
		break;
	case LayerType::IMAGE:
		// Do nothing.
		break;
	}
}

// Inference.
void RunNetwork(NeuralNetwork &network) {
	for (int i = 1; i < network.layers.size(); i++) {
		ApplyLayer(*network.layers[i], network.layers[i - 1]->neurons);
	}
}

void GetLabel(NeuralLayer &layer) {
	float maxValue = -INFINITY;
	int label = -1;
	for (int i = 0; i < layer.numNeurons; i++) {
		if (layer.neurons[i] > maxValue) {
			label = i;
			maxValue = layer.neurons[i];
		}
	}
}

void InitializeNetwork(NeuralNetwork &network) {
	for (int i = 1; i < network.layers.size(); i++) {
		NeuralLayer &layer = *network.layers[i];
		assert(layer.numInputs == network.layers[i - 1]->numNeurons);
		layer.neurons = new float[layer.numNeurons];
		switch (layer.type) {
		case LayerType::FC:
			layer.biases = new float[layer.numNeurons]{};
			layer.weights = new float[layer.numNeurons * layer.numInputs]{};
			break;
		case LayerType::RELU:
			assert(layer.numNeurons == layer.numInputs);
			break;
		case LayerType::IMAGE:
			break;
		}
	}
}

float ComputeSVMLoss(const float *data, size_t size, int correctLabel) {
	float loss = 0.0f;
	float truth = data[correctLabel];
	for (size_t i = 0; i < size; i++) {
		if (i == truth)
			continue;
		loss += std::max(0.0f, data[i] - truth + 1.0f);
	}
	return loss;
}

float ComputeSoftMaxLoss(const float *data, size_t size, int correctLabel) {
	float sumExp = 0.0f;
	for (size_t i = 0; i < size; i++) {
		float exped = expf(data[i]);
		sumExp += exped;
	}
	float normalized = expf(data[correctLabel]) / sumExp;
	return -logf(normalized);
}

/*
Tensor LoadImageAsTensor(std::string path, bool monochrome) {
	FILE *f = fopen(path.c_str(), "rb");
	int w, h, comp;
	stbi_uc *data = stbi_load_from_file(f, &w, &h, &comp, 3);

	// TODO: Load all three channels.
	Tensor tensor(w, h, 1);
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			tensor.At(x, y, 0) = ByteToFloat(data[(y * w + h) * 3]);
		}
	}
}*/

std::vector<DataVector> LoadMNISTImages(std::string path) {
	FILE *f = fopen(path.c_str(), "rb");
	if (!f)
		throw;
	uint32_t magic = readBE32(f);
	if (magic != 0x803) {
		return std::vector<DataVector>();
	}
	int imageCount = readBE32(f);
	std::vector<DataVector> images(imageCount);
	int rows = readBE32(f);
	int cols = readBE32(f);
	uint8_t *temp = new uint8_t[rows * cols];
	for (int i = 0; i < imageCount; i++) {
		DataVector &image = images[i];
		fread(temp, 1, rows*cols, f);
		image.data = new float[rows * cols];
		for (int j = 0; j < rows*cols; j++) {
			image.data[j] = ByteToFloat(temp[j]);
		}
		image.size = rows * cols;
		image.dim = { cols, rows, 1 };
	}
	delete[] temp;
	fclose(f);
	return images;
}

std::vector<uint8_t> LoadMNISTLabels(std::string path) {
	FILE *f = fopen(path.c_str(), "rb");
	if (!f)
		throw;
	uint32_t magic = readBE32(f);
	uint32_t count = readBE32(f);
	if (magic != 0x801) {
		return std::vector<uint8_t>();
	}
	std::vector<uint8_t> data(count);
	fread(data.data(), 1, count, f);
	fclose(f);
	return data;
}

int main() {
	float data[3] = { 3.2, 5.1, -1.7 };
	float loss = ComputeSoftMaxLoss(data, 3, 0);

	auto trainImages = LoadMNISTImages("C:/dev/MNIST/train-images.idx3-ubyte");
	auto trainLabels = LoadMNISTLabels("C:/dev/MNIST/train-labels.idx1-ubyte");
	assert(trainImages.size() == trainLabels.size());

	auto testImages = LoadMNISTImages("C:/dev/MNIST/train-images.idx3-ubyte");
	auto testLabels = LoadMNISTLabels("C:/dev/MNIST/train-labels.idx1-ubyte");
	assert(testImages.size() == testLabels.size());

	NeuralNetwork network;
	NeuralLayer imageLayer{ LayerType::IMAGE, ivec3{ 28, 28, 1 } };
	imageLayer.numInputs = 0;
	imageLayer.numNeurons = 28 * 28;

	NeuralLayer hiddenLayer{ LayerType::FC, ivec3{100,1,1} };
	hiddenLayer.numInputs = 28 * 28;
	hiddenLayer.numNeurons = 100;

	NeuralLayer relu{ LayerType::RELU };
	relu.numInputs = 100;
	relu.numNeurons = 100;

	NeuralLayer fcLayer{ LayerType::FC, ivec3{ 32,32,1 } };
	fcLayer.numInputs = 100;
	fcLayer.numNeurons = 10;

	network.layers.push_back(&imageLayer);
	network.layers.push_back(&hiddenLayer);
	network.layers.push_back(&relu);
	network.layers.push_back(&fcLayer);
	network.lossFunction = &ComputeSVMLoss;

	InitializeNetwork(network);

	static const char *labelNames[10] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

	// Evaluate network on the full training set.
	float totalLoss = 0.0f;
	for (int i = 0; i < trainImages.size(); i++) {
		assert(imageLayer.numNeurons == trainImages[i].size);
		imageLayer.neurons = trainImages[i].data;
		RunNetwork(network);
		float loss = network.lossFunction(fcLayer.neurons, 10, trainLabels[i]);
		totalLoss += loss;
	}
	totalLoss /= trainLabels.size( );

	printf("Total loss: %0.1f\n", totalLoss);
	
	// Test
	for (int i = 0; i < testImages.size(); i++) {

	}
	return 0;
}

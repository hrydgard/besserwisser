#include <cstdio>
#include <vector>

#include "math_util.h"

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
	double sum = 0.0;
	for (int i = 0; i < imageCount; i++) {
		DataVector &image = images[i];
		fread(temp, 1, rows*cols, f);
		image.data = new float[rows * cols + 1];
		for (int j = 0; j < rows*cols; j++) {
			image.data[j] = ByteToFloat(temp[j]);
			sum += image.data[j];
		}
		image.data[rows * cols] = 1.0f;  // Bias trick
		image.size = rows * cols + 1;
		image.dim = { cols, rows, 1 };
	}
	delete[] temp;
	fclose(f);

	sum /= (double)(rows * cols * imageCount);
	float sumFloat = (float)sum;

	// Go back and renormalize all the image data around 0.0f;
	for (int i = 0; i < imageCount; i++) {
		DataVector &image = images[i];
		for (int j = 0; j < rows * cols; j++) {   // Take care not to rescale the bias trick float
			image.data[j] -= sumFloat;
		}
	}
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

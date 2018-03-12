#include <cstdio>
#include <vector>

#include "math_util.h"
#include "blob.h"
#include "mnist.h"

std::vector<Blob> LoadMNISTImages(std::string path) {
	FILE *f = fopen(path.c_str(), "rb");
	if (!f) {
		fprintf(stderr, "ERROR: Could not open '%s'\n", path.c_str());
		return std::vector<Blob>();
	}
	uint32_t magic = readBE32(f);
	if (magic != 0x803) {
		fprintf(stderr, "Bad file format %08x (%s)\n", magic, path.c_str());
		return std::vector<Blob>();
	}
	int imageCount = readBE32(f);
	std::vector<Blob> images(imageCount);
	int rows = readBE32(f);
	int cols = readBE32(f);
	double sum = 0.0;
	for (int i = 0; i < imageCount; i++) {
		Blob &image = images[i];
		image.data = new uint8_t[rows * cols + 1];
		fread(image.data, 1, rows*cols, f);
		image.type = DataType::UINT8_T_SCALED;
		image.scale = 1.0f / 255.0f;
		image.offset = 0.0f;
		((uint8_t *)image.data)[rows * cols] = 255;  // Bias trick
		image.size = rows * cols + 1;
		image.dim = { 1, cols, rows };
	}
	fclose(f);
	return images;
}

std::vector<uint8_t> LoadMNISTLabels(std::string path) {
	FILE *f = fopen(path.c_str(), "rb");
	if (!f) {
		fprintf(stderr, "ERROR: Could not open '%s'\n", path.c_str());
		return std::vector<uint8_t>();
	}
	uint32_t magic = readBE32(f);
	uint32_t count = readBE32(f);
	if (magic != 0x801) {
		fprintf(stderr, "Bad file format %08x (%s)\n", magic, path.c_str());
		return std::vector<uint8_t>();
	}
	std::vector<uint8_t> data(count);
	fread(data.data(), 1, count, f);
	fclose(f);
	return data;
}

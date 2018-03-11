#pragma once

#include <cassert>

struct Dim {
	int count = 1;
	int depth, height, width;

	int TotalSize() const {
		return count * depth * height * width;
	}
	int GetIndex(int x, int y, int z, int n = 0) const {
		return n * (depth * width * height) + (width * height * z) + (width * y) + x;
	}
};

enum class DataType {
	UINT8_T_SCALED,
	FLOAT32,
};

struct Blob {
	~Blob() {
		switch (type) {
		case DataType::FLOAT32:
			delete[](float*)data;
		default:
			assert(false);
			break;
		}
	}
	const float *GetFloatPtr() const {
		assert(type == DataType::FLOAT32);
		return (const float *)data;
	}
	void CopyToFloat(float *output) const;

	DataType type = DataType::FLOAT32;
	void *data = nullptr;
	size_t size = 0;
	Dim dim;

	// For quantized data.
	float scale = 1.0f;
	float offset = 0.0f;
};

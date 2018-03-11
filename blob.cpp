#include <cstring>
#include <cstdint>

#include "blob.h"
#include "math_util.h"

void Blob::CopyToFloat(float *output) const {
	switch (type) {
	case DataType::FLOAT32:
		memcpy(output, data, sizeof(float) * size);
		break;
	case DataType::UINT8_T_SCALED:
	{
		// TODO: SIMD-optimize.
		BytesToFloat(output, (const uint8_t *)data, size, scale, offset);
		break;
	}
	}
}

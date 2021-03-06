#ifndef MP_HANDLER_HPP
#define MP_HANDLER_HPP

#include <vector>
#include <msgpack.hpp>

#include "file_handler.hpp"

namespace knn_finder {

struct item_record {
	unsigned int idx;
	std::vector<float> fvec;

	MSGPACK_DEFINE(idx, fvec)
};

int read_mp(const char *filename, bool is_gzip, float **mat_ptr, int *nr_ptr, int *nc_ptr);

} // knn_finder

#endif // MP_HANDLER_HPP

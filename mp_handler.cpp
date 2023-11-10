#include <iostream>
#include <iterator>
#include <vector>
#include <msgpack.hpp>

#include "mp_handler.hpp"

namespace knn_finder {

int read_mp(const char *filename, bool is_gzip, float **mat_ptr, int *nr_ptr, int *nc_ptr) {
	std::vector<item_record> records;
	{ // Block to destruct the file read buffer after copied the content to the records.
		file_reader reader(is_gzip);
		try{
			reader.open(filename, true);
		}
		catch (const std::exception& e){
			return FILE_OPEN_FAILED;
		}
		std::istream& is = reader.get();

		// Seemingly unnecessary parentheses surrounding the 1st parameter is to stop
		// the compiler to treat this statement as a function declaration.
		// See "most vexing parse".
		std::vector<char> buf((std::istreambuf_iterator<char>(is)),
			std::istreambuf_iterator<char>());

		std::size_t offset = 0;
		while (offset < buf.size()) {
			msgpack::object_handle msg;
			msgpack::unpack(msg, &buf[0], buf.size(), offset);
			msgpack::object obj(msg.get());
			item_record record;
			obj.convert(record);
			records.push_back(record);
		}
	}

	int nr = records.size(), nc = records.at(0).fvec.size();
	*nc_ptr = nc, *nr_ptr = nr;

	*mat_ptr = (float *)malloc(sizeof(float) * nr * nc);
	if (*mat_ptr == NULL) return MALLOC_FAILED;

	for (int i = 0; i < nr; i++) {
		int idx = records[i].idx;
		if (idx >= nr) return INDEX_OUT_OF_RANGE;

		for (int j = 0; j < nc; j++) {
			(*mat_ptr)[j*nr+idx] = records[i].fvec[j];
		}
	}

	return 0;
}

} // knn_finder

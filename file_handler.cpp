#include <iostream>

#include "file_handler.hpp"

namespace knn_finder {

file_reader::file_reader(bool is_gzip) : ifs(), decomp(), sbuf(), is(&sbuf) {
	if (is_gzip) sbuf.push(decomp);
	sbuf.push(ifs);
}

bool file_reader::open(const char *filename, bool is_binary) {
	std::ios_base::openmode mode = std::ios_base::in;
	if (is_binary) mode |= std::ios_base::binary;
	ifs.open(filename, mode);
	return ifs.is_open();
}

std::istream& file_reader::get() {
	return is;
}

void check_error(int res) {
	switch (res) {
		case 0:
			return;
		case INVALID_ARGUMENT:
			std::cerr << "Invalid argument." << std::endl;
			break;
		case FILE_OPEN_FAILED:
			std::cerr << "Could not open file." << std::endl;
			break;
		case EMPTY_INPUT:
			std::cerr << "Input file is empty, or failed to decode it." << std::endl;
			break;
		case INVALID_FORMAT:
			std::cerr << "Invalid file format." << std::endl;
			break;
		case MALLOC_FAILED:
			std::cerr << "Could not allocate memory." << std::endl;
			break;
		case INDEX_OUT_OF_RANGE:
			std::cerr << "Record index is out of range." << std::endl;
			break;
		default:
			std::cerr << "Unknown error." << std::endl;
	}
	exit(res);
}

} // knn_finder

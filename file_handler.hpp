#ifndef FILE_HANDLER_HPP
#define FILE_HANDLER_HPP

#include <iostream>
#include <fstream>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>

#define INVALID_ARGUMENT 1
#define FILE_OPEN_FAILED 2
#define EMPTY_INPUT 3
#define INVALID_FORMAT 4
#define MALLOC_FAILED 5
#define INDEX_OUT_OF_RANGE 6

class file_reader {
public:
	file_reader(bool is_gzip=false);

	bool open(const char *filename, bool is_binary=false);
	std::istream& get();

private:
	std::ifstream ifs;
	boost::iostreams::gzip_decompressor decomp;
	boost::iostreams::filtering_streambuf<boost::iostreams::input> sbuf;
	std::istream is;
};

void check_error(int res);

#endif // FILE_HANDLER_HPP

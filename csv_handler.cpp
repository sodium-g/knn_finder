#include <iostream>
#include <cstdlib>
#include <boost/tokenizer.hpp>
#include <boost/regex.hpp>

#include "csv_handler.hpp"

namespace knn_finder {

bool is_numeric(const std::string& str) {
	static const boost::regex ex("[+-]?\\d+(\\.\\d+)?([Ee][+-]?\\d+)?");
	return boost::regex_match(str, ex);
}

int read_csv(const char *filename, bool is_gzip, float **mat_ptr, int *nr_ptr, int *nc_ptr) {
	file_reader reader(is_gzip);
	try{
		reader.open(filename, is_gzip);
	}catch(const std::exception& e){
		return FILE_OPEN_FAILED;
	}
	std::istream& is = reader.get();

	int nr = 0, nc = 0;
	std::string line;
	std::vector<float> vec;
	boost::char_separator<char> sep(IN_SEP);

	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	while (std::getline(is, line)) {
		tokenizer tok(line, sep);
		int i = 0;
		for (tokenizer::iterator it = tok.begin(); it != tok.end(); it++) {
			if (!is_numeric(*it)) return INVALID_FORMAT;
			vec.push_back(atof(it->c_str()));
			if (++i == nc) break;
		}
		if (nr == 0 && !(nc = i)) return EMPTY_INPUT;
		for (; i < nc; i++) vec.push_back(0);
		nr++;
	}
	if (nr == 0) return EMPTY_INPUT;
	*nc_ptr = nc, *nr_ptr = nr;

	*mat_ptr = (float *)malloc(sizeof(float) * nr * nc);
	if (*mat_ptr == NULL) return MALLOC_FAILED;

	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nc; j++) {
			(*mat_ptr)[j*nr+i] = vec[i*nc+j];
		}
	}

	return 0;
}

} // knn_finder

#ifndef CSV_HANDLER_HPP
#define CSV_HANDLER_HPP

#include <fstream>

#include "file_handler.hpp"

#ifndef IN_SEP
#define IN_SEP " \t,"
#endif
#ifndef OUT_SEP
#define OUT_SEP "\t"
#endif

namespace knn_finder {

int read_csv(const char *filename, bool is_gzip, float **mat_ptr, int *nr_ptr, int *nc_ptr);

template<typename T>
int write_csv(const char *filename, const T *mat, int nr, int nc) {
	std::ofstream ofs(filename);
	if (!ofs) return FILE_OPEN_FAILED;

	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nc; j++) {
			ofs << mat[j*nr+i];
			if (j < nc - 1) ofs << OUT_SEP;
		}
		ofs << std::endl;
	}

	return 0;
}

} // knn_finder

#endif // CSV_HANDLER_HPP

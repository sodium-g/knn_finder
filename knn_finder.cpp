#include <iostream>
#include <boost/program_options.hpp>

#include "csv_handler.hpp"
#include "mp_handler.hpp"
#include "knn_cuda.hpp"

#ifndef DIST_EXT
#define DIST_EXT ".dist"
#endif
#ifndef IDX_EXT
#define IDX_EXT ".idx"
#endif
#ifndef K
#define K 100
#endif

using namespace knn_finder;

int main(int argc, char **argv) {
	namespace po = boost::program_options;

	// required() function was not available until Boost 1.42, so we do not use it here.
	po::options_description desc("Available options");
	desc.add_options()
		("help,h", "display usage")
		("infile,i", po::value<std::string>(), "input filename (mandatory)")
		("outfile,o", po::value<std::string>(), "basename (wo extension) of output files")
		("in-pack,p", "input file is in MessagePack format (CSV text otherwise)")
		("in-gzip,z", "input file is gzip compressed")
		("neigh,k", po::value<int>(), "nearest neighbours to find")
		("sparse,s", "input matrix is sparse")
	;

	po::variables_map vm;
	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	} catch (po::error& ex) {
		std::cerr << ex.what() << std::endl;
		std::cerr << desc;
		return INVALID_ARGUMENT;
	}

	if (vm.count("help") || !vm.count("infile")) {
		std::cerr << desc;
		return INVALID_ARGUMENT;
	}

	const std::string in_filename = vm["infile"].as<std::string>();
	std::string out_basename = in_filename;
	if (vm.count("outfile")) out_basename = vm["outfile"].as<std::string>();
	const std::string dist_filename = out_basename + DIST_EXT;
	const std::string idx_filename = out_basename + IDX_EXT;

	int k = K;
	if (vm.count("neigh")) k = vm["neigh"].as<int>();

	float *mat; // Input records, in column-major order.
	int mat_width, mat_height;
	std::cout << "Reading " << in_filename << std::endl;
	if (vm.count("in-pack")) {
		check_error(read_mp(in_filename.c_str(), vm.count("in-gzip"), &mat, &mat_width, &mat_height));
	} else {
		check_error(read_csv(in_filename.c_str(), vm.count("in-gzip"), &mat, &mat_width, &mat_height));
	}
	std::cout << "Done." << std::endl;

	float *dist; // Distance records to be generated, in column-major order.
	int *idx; // Index records to be generated, in column-major order.
	dist = (float *)malloc(mat_width * k * sizeof(float));
	idx = (int *)malloc(mat_width * k * sizeof(int));
	if (dist == NULL || idx == NULL) check_error(MALLOC_FAILED);

	std::cout << std::endl << "Gathering device information and settings..." << std::endl;
	print_props();
	std::cout << "Number of records: " << mat_width << std::endl;
	std::cout << "Dimension of records: " << mat_height << std::endl;
	std::cout << "Nearest neighbours to find: " << k << std::endl;

	if (mat_width < k) {
		std::cerr << "Currently, number of records less than k is not supported." << std::endl;
		return EMPTY_INPUT;
	}

	std::cout << std::endl << "Searching k-nearest neighbours..." << std::endl;
	find_knn(mat, mat_width, mat_height, k, dist, idx, vm.count("sparse"));

	std::cout << std::endl << "Writing search result (distance) to " << dist_filename << std::endl;
	check_error(write_csv<float>(dist_filename.c_str(), dist, mat_width, k));
	std::cout << "Writing search result (index) to " << idx_filename << std::endl;
	check_error(write_csv<int>(idx_filename.c_str(), idx, mat_width, k));
	std::cout << "Done." << std::endl;

	free(idx);
	free(dist);
	free(mat);

	return 0;
}

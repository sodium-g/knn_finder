knn_finder
==========

a brute force k-nearest neighbour search program for nVidia GPUs


## Requirements and tested versions
- gcc, tested with 13.2.1
- cuda, tested with 12.3.0
- boost, tested with 1.83.0
- msgpack >= 2.0.0, tested with 5.0.0

### Note
If you are using the above versions, use Makefile.cuda12gcc13 instead of Makefile to make nvcc use an appropriate version of gcc.
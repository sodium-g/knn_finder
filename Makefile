CXX = /usr/bin/g++
NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -O2
LDFLAGS =
LDLIBS = -lcuda -lcublas -lboost_regex -lboost_iostreams -lboost_program_options -lmsgpackc
BUILD_DIR = build
PREFIX = /usr/local

HANDLERS = file_handler csv_handler mp_handler
KERNELS = knn_cuda
HANDLER_OBJS = $(HANDLERS:%=$(BUILD_DIR)/%.o)
KERNEL_OBJS = $(KERNELS:%=$(BUILD_DIR)/%.o)
OBJS = $(HANDLER_OBJS) $(KERNEL_OBJS) $(BUILD_DIR)/knn_finder.o
HEADERS = $(HANDLERS:%=%.hpp) $(KERNELS:%=%.hpp)

.PHONY: all install clean

all: $(BUILD_DIR)/knn_finder

install:
	mkdir -p $(PREFIX)/bin
	cp $(BUILD_DIR)/knn_finder $(PREFIX)/bin

clean:
	rm -f $(BUILD_DIR)/knn_finder
	rm -f $(BUILD_DIR)/*.o

$(BUILD_DIR)/knn_finder: $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(HANDLER_OBJS): $(BUILD_DIR)/%.o: %.cpp %.hpp
	$(CXX) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/csv_handler.o $(BUILD_DIR)/mp_handler.o: file_handler.hpp

$(KERNEL_OBJS): $(BUILD_DIR)/%.o: %.cu %.hpp
	$(NVCC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR)/knn_finder.o: knn_finder.cpp $(HEADERS)
	$(NVCC) $(CFLAGS) -c -o $@ $<

$(OBJS): | $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

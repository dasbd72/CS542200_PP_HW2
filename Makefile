# CC = gcc
# CXX = g++
CC = clang
CXX = clang++
LDLIBS = -lpng
CFLAGS = -lm -O3
CFLAGS += -Wall -Wextra
CFLAGS += -march=native
hw2a: CFLAGS += -pthread -Ofast
hw2b: CC := mpicc -cc=$(CC)
hw2b: CXX := mpicxx -cxx=$(CXX)
hw2b: CFLAGS += -fopenmp -pthread -Ofast
CXXFLAGS = -std=c++17 $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)

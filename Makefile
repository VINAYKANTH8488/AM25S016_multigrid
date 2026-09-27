CXX      := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -Iinclude
SRC      := src/poisson.cpp src/smoothers.cpp src/stencil.cpp src/multigrid.cpp src/main.cpp
OBJ      := $(SRC:.cpp=.o)
BIN      := multigrid

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SRC)

clean:
	rm -f $(BIN) results/*.csv

run: all
	./$(BIN)

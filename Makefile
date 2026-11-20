# Makefile for Experiment 6: Memory Optimization in CUDA
# Targets Linux / Colab / macOS (with CUDA toolkit)

NVCC := nvcc
NVCCFLAGS := -O2 -std=c++17
SRC := src/exp6.cu
BIN_DIR := bin
TARGET := $(BIN_DIR)/exp6

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@
	@cp $(TARGET) exp6 2>/dev/null || true
	@echo "Build successful: $(TARGET)"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BIN_DIR) exp6 exp6.exe *.o *.obj

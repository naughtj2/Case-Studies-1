# Compiler and flags
CC = mpicc
CFLAGS =  -O2
LDFLAGS = -llapacke -llapack -lblas -lm

# Targets
TARGETS = TSQR_Q2 TSQR_Q3

# Source files
SRC_Q2 = TSQR_Q2.c
SRC_Q3 = TSQR_Q3.c
PYTHON_SCRIPT = TSQR_Q3.py

# Executables
EXE_Q2 = TSQR_Q2
EXE_Q3 = TSQR_Q3

all: $(TARGETS)

$(EXE_Q2): $(SRC_Q2)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(EXE_Q3): $(SRC_Q3)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run_python:
	python3 $(PYTHON_SCRIPT)

clean:
	rm -f $(TARGETS) *.o MScaling.png NScaling.png

.PHONY: all clean run_q3


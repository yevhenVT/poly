CC = g++
VPATH = ../../../
CFLAGS = -std=c++0x -g -w
LDFLAGS = -lconfig++
objects = poisson_noise.o HH2_buffer.o main.o

all: $(objects)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o out

main.o: main.cpp HH2_buffer.h poisson_noise.h
	$(CC) $(CFLAGS) -c $< -o $@

HH2_buffer.o: HH2_buffer.cpp HH2_buffer.h exception.h poisson_noise.h
	$(CC) $(CFLAGS) -c $< -o $@

poisson_noise.o: poisson_noise.cpp poisson_noise.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o

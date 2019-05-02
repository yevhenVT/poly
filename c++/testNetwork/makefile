CC = mpic++
VPATH = ../src/
CFLAGS = -std=c++0x -g -w -O2 -I../include
objects = utils.o HH2.o HHI.o poisson_noise.o HvcNetwork.o main.o

all: $(objects)
	$(CC) $(CFLAGS) $^ -o testNetwork

main.o: main.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

HvcNetwork.o: HvcNetwork.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

HH2.o: HH2.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

HHI.o: HHI.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

poisson_noise.o: poisson_noise.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

utils.o: utils.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	$(RM) count *.o *~

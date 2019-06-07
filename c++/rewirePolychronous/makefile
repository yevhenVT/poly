CC = mpic++
VPATH = ../src/
CFLAGS = -std=c++0x -g -w -O2 -I../include
objects = utils.o poisson_noise.o HHI.o HH2.o HvcNetwork.o main.o

all: $(objects)
	$(CC) $(CFLAGS) $^ -o rewire

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

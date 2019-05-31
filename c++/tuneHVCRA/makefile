CC = mpic++
VPATH = ../src
CFLAGS = -std=c++0x -g -w -O2 -I../include
objects = poisson_noise.o HH2.o main.o

all: $(objects)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o exec

main.o: main.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

HH2.o: HH2.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

poisson_noise.o: poisson_noise.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	$(RM) count *.o *~

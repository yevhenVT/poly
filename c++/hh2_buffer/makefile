CC = mpic++
VPATH = ../src/
CFLAGS = -std=c++0x -g -w -I../include/
objects = poisson_noise.o HH2.o main.o

all: $(objects)
	$(CC) $(CFLAGS) $^ -o out

main.o: main.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

HH2.o: HH2.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

poisson_noise.o: poisson_noise.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	$(RM) count *.o *~

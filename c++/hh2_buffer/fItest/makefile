CC = mpic++
VPATH = ../../../
CFLAGS = -std=c++0x -g -w
LDFLAGS = -lconfig++
objects = poisson_noise.o HH2_buffer.o main.o

all: $(objects)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o out

main.o: main.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

HH2_buffer.o: HH2_buffer.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

poisson_noise.o: poisson_noise.cpp
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	$(RM) count *.o *~

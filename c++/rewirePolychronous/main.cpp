#include "HvcNetwork.h"
#include "utils.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>

int main(int argc, char** argv)
{
	int rank; // MPI process rank
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	double fraction = 0.7; // fraction of connections to rewire
    std::string outputDir = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.7/"; // directory to which write network information
	std::string networkDir = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/"; // directory with original network
	
	unsigned seed = 1991; // seed for random number generators
	
	HvcNetwork hvc(seed);
	
	hvc.rewire_fraction_connections(fraction, networkDir, outputDir);
	
	MPI_Finalize();
	
	return 0;

}

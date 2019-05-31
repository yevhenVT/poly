#include "HvcNetwork.h"
#include "utils.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>

static void write_training_neurons(int num_training, const std::string& outDir){
	// sample training neurons
	std::vector<int> training_neurons(num_training);
	std::iota(training_neurons.begin(), training_neurons.end(), 0);
	
	
	std::ofstream out;
  
	out.open((outDir+"training_neurons.bin").c_str(), std::ios::out | std::ios::binary);

	out.write(reinterpret_cast<char *>(&num_training), sizeof(num_training));
	
				 
	for (size_t i = 0; i < num_training; i++)
		out.write(reinterpret_cast<char *>(&training_neurons[i]), sizeof(training_neurons[i]));
	
	out.close();
}

int main(int argc, char** argv)
{
	int rank; // MPI process rank
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int N_RA = 20000; // number of HVC(RA) neurons
	int N_I = 0; // number of HVC(I) neurons
	double pie = 0.0; // probability to contact an HVC-RA neuron by interneuron
	double pei = 0.0; // probability to contact an HVC-I neuron by HVC-RA neuron
	
    std::string outputDirectory = "/home/eugene/Programming/data/mlong/randomFeedforward/random/network/"; // directory to which write network information
	
	double Gee_max = 0.004; // max strength of HVC-RA -> HVC-RA connection
	double Gie_max = 0.0;  // max strength of HVC-I -> HVC-RA connection
	double Gei_max = 0.0;  // max strength of HVC-RA -> HVC-I connection
	
	double mean_delay = 0.0; // mean axonal conductance delay for HVC-RA -> HVC-RA
	double sd_delay = 0.0; // standard deviation of axonal conductance delay for HVC-RA -> HVC-RA
	
	int num_output = 170; // number of output connections (same as number of neurons in each layer for synfire chain)
	
	unsigned seed = 1991; // seed for random number generators
	
	HvcNetwork hvc(seed);
	
	hvc.set_connection_strengths(Gee_max, Gei_max, Gie_max);
	hvc.sample_network_without_RA2RA(N_RA, N_I, pei, pie, mean_delay, sd_delay);
		
	if (rank == 0){
		write_training_neurons(num_output, outputDirectory);
	}
	
	hvc.wire_random_feedforward(num_output, mean_delay, sd_delay, outputDirectory);
	
	MPI_Finalize();
	
	return 0;

}

#include "HvcNetwork.h"
#include "utils.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>

static void print_usage(){
	std::cout << "Usage: 4 different types of network can be wired by specifying one of the options below:\n"
			  << "--synfire to wire one or more synfire chains\n"
			  << "--polychronous to wire a polychronous network\n"
			  << "--integration_times_uniform to wire a polychronous network using distributed integration times from a uniform distribution\n"
			  << "--integration_times_lognormal to wire a polychronous network using distributed integration times from a truncated lognormal distribution\n\n"
			  
			  << "Common parameters that have to specified for all networks:\n"
			  << "-o <output directory>\n"
			  << "-nra <# of HVC-RA neurons>\n"
			  << "-nout <# of output connections to other HVC-RA for each HVC-RA>\n"
			  << "-gee <max synaptic strength for HVC-RA -> HVC-RA connections>\n\n"
		      
		      << "Common parameters that have to specified for synfire chain and polychronous networks:\n"
		      << "-md <mean value for axonal conduction delay>\n"
			  << "-sd <standard deviation of axonal conduction delays>\n\n"
			  
			  << "Parameters that have to specified for polychronous network with distributed integration times sampled from a uniform distribution:\n"
		      << "-c_max <max dendritic capacitance>\n"
			  << "-c_min <min dendritic capacitance>\n\n"
			  
			  << "Parameters that have to specified for polychronous network with distributed integration times sampled from a truncated lognormal distribution:\n"
		      << "-c_mean <mean dendritic capacitance>\n"
		      << "-c_std <std dendritic capacitance>\n"
			  << "-c_min <min dendritic capacitance>\n\n"
			  
			  << "User can also define an interneuron population:\n" 
			  << "-ni <# of HVC-I neurons>\n"
			  << "-gei <max synaptic strength for HVC-RA -> HVC-I connections>\n"
			  << "-gie <max synaptic strength for HVC-I -> HVC-I connections>\n"
			  << "-pei <probability to contact an HVC-I neuron for HVC-RA neuron>\n"
			  << "-pie <probability to contact an HVC-RA neuron for interneuron>\n\n"
			  
			  << "User can provide a seed for random number generators:\n" 
			  << "-seed <non-negative number>\n\n"
			  
			  << "In polychronous networks inputs have to arrive synchronously in a small time window.\n"
			  << "User can define the time window size and maximal number of input connections each neuron receives\n"
			  << "Parameters that have to specified for polychronous networks:\n"
			  << "-sm <window size for synchronously arriving inputs>\n"
			  << "-maxnin <maximal number of input connections per HVC-RA neuron>\n\n"
			  
			  << "Parameters that have to specified for synfire chain networks:\n"
			  << "-npsc <number of parallel synfire chains>\n"
			  
			  << "\nTo see the usage message, type --help:\n\n"
			  << std::endl;	
}



static int parse_command_line(int argc, char** argv, int rank, bool& s, bool& p, bool& int_uniform, bool& int_lognormal,
						std::string& outDir, int& Nra, int& Ni, unsigned& seed, int& nout, int& maxnin, int& npsc,
						double& gee, double& gei, double& gie, double& pei, double& pie, double& sm, double& md, double& sd, 
						std::pair<double,double>& int_range, double& int_mean, double& int_std){
	if (argc > 1){
		// check the network option and make sure that only one option is selected
		int num_true_indicators = 0;
       
        if (cmdOptionExist(argv+1, argv+argc, "--synfire")) {s = true; num_true_indicators+= 1;};
        if (cmdOptionExist(argv+1, argv+argc, "--polychronous")) {p = true; num_true_indicators+= 1;};
        if (cmdOptionExist(argv+1, argv+argc, "--integration_times_uniform")) {int_uniform = true; num_true_indicators+= 1;};
        if (cmdOptionExist(argv+1, argv+argc, "--integration_times_lognormal")) {int_lognormal = true; num_true_indicators+= 1;};
        
        
        if ( (num_true_indicators == 0) || (num_true_indicators > 1)){
			if (rank == 0) print_usage();
			return -1;
		}
		
		// read common information:
		// output directory; number of HVC-RA neurons; strength of HVC-RA -> HVC-RA connections;
		// number of outputs for HVC-RA
		if (cmdOptionExist(argv+1, argv+argc, "-o")) outDir = cmdOptionParser(argv+1, argv+argc, "-o");
        else {
			if (rank == 0) {std::cout << "Output directory -o is missing!\n" <<std::endl; print_usage();}	
			return -1;	
		}
		
        if (cmdOptionExist(argv+1, argv+argc, "-nra")) Nra = atoi(cmdOptionParser(argv+1, argv+argc, "-nra"));
        else {
			if (rank == 0) {std::cout << "Number of HVC-RA -nra is missing!\n" <<std::endl; print_usage();}
			return -1;	
		}
		
		if (cmdOptionExist(argv+1, argv+argc, "-nout")) nout = atoi(cmdOptionParser(argv+1, argv+argc, "-nout"));
        else {
			if (rank == 0){std::cout << "Number of HVC-RA outputs -nout is missing!\n" <<std::endl; print_usage();}
			return -1;	
		}
		
		if (cmdOptionExist(argv+1, argv+argc, "-gee")) gee = atof(cmdOptionParser(argv+1, argv+argc, "-gee"));
        else {
			if (rank == 0){std::cout << "Strength of HVC-RA -> HVC-RA connection -gee is missing!\n" <<std::endl; print_usage();}
			return -1;	
		}

		if (p || s){
			if (cmdOptionExist(argv+1, argv+argc, "-md")) md = atof(cmdOptionParser(argv+1, argv+argc, "-md"));
			else {
				if (rank == 0){std::cout << "Mean axonal conduction delay for HVC-RA -> HVC-RA -md is missing!\n" <<std::endl; print_usage();}
				return -1;	
			}
			
			if (cmdOptionExist(argv+1, argv+argc, "-sd")) sd = atof(cmdOptionParser(argv+1, argv+argc, "-sd"));
			else {
				if (rank == 0){std::cout << "Std axonal conduction delay for HVC-RA -> HVC-RA -sd is missing!\n" << std::endl; print_usage();}
				return -1;	
			}		
		}
			
		// if network is polychronous, read information about max number of input connections and synchronous margin:
		// num interneurons and strength of connection between HVC-RA and HVC-I
		if (p || int_uniform || int_lognormal){
			if (cmdOptionExist(argv+1, argv+argc, "-maxnin")) maxnin = atoi(cmdOptionParser(argv+1, argv+argc, "-maxnin"));
			else {
				if (rank == 0){std::cout << "Max number of input for HVC-RA -maxnin is missing!\n" << std::endl; print_usage();}
				return -1;	
			}
       
			if (cmdOptionExist(argv+1, argv+argc, "-sm")) sm = atof(cmdOptionParser(argv+1, argv+argc, "-sm"));
			else {
				if (rank == 0){std::cout << "Size of synchronous window for inputs arrival -sm is missing!\n" << std::endl; print_usage();}
				return -1;	
			}
		}
		
		// if network is synfire chain, read the number of synfire chains to wire
		if (s){
			if (cmdOptionExist(argv+1, argv+argc, "-npsc")) npsc = atoi(cmdOptionParser(argv+1, argv+argc, "-npsc"));
			else {
				if (rank == 0){std::cout << "Number of parallel chains -npsc is missing!\n" << std::endl; print_usage();}
				return -1;	
			}
		}
		if (int_uniform || int_lognormal){
			if (cmdOptionExist(argv+1, argv+argc, "-int_min")) int_range.first = atof(cmdOptionParser(argv+1, argv+argc, "-int_min"));
			else {
				if (rank == 0){std::cout << "min integration time -int_min is missing!\n" << std::endl; print_usage();}
				return -1;	
			}
			
			if (cmdOptionExist(argv+1, argv+argc, "-int_max")) int_range.second = atof(cmdOptionParser(argv+1, argv+argc, "-int_max"));
			else {
				if (rank == 0){std::cout << "max integration time -int_max is missing!\n" << std::endl; print_usage();}
				return -1;	
			}
		}
		
		if (int_lognormal){	
			if (cmdOptionExist(argv+1, argv+argc, "-int_mean")) int_mean = atof(cmdOptionParser(argv+1, argv+argc, "-int_mean"));
			else {
				if (rank == 0){std::cout << "mean integration time -int_mean is missing!\n" << std::endl; print_usage();}
				return -1;	
			}
			
			if (cmdOptionExist(argv+1, argv+argc, "-int_std")) int_std = atof(cmdOptionParser(argv+1, argv+argc, "-int_std"));
			else {
				if (rank == 0){std::cout << "std integration time -int_std is missing!\n" << std::endl; print_usage();}
				return -1;	
			}
		}
		
		
		// num interneurons and strength of connection between HVC-RA and HVC-I
		if (cmdOptionExist(argv+1, argv+argc, "-ni")) Ni = atoi(cmdOptionParser(argv+1, argv+argc, "-ni"));		
		if (cmdOptionExist(argv+1, argv+argc, "-gie")) gie = atof(cmdOptionParser(argv+1, argv+argc, "-gie"));
		if (cmdOptionExist(argv+1, argv+argc, "-gei")) gei = atof(cmdOptionParser(argv+1, argv+argc, "-gei"));
		if (cmdOptionExist(argv+1, argv+argc, "-pei")) gei = atof(cmdOptionParser(argv+1, argv+argc, "-pei"));
		if (cmdOptionExist(argv+1, argv+argc, "-pie")) gei = atof(cmdOptionParser(argv+1, argv+argc, "-pie"));
		
		char* endptr;
		if (cmdOptionExist(argv+1, argv+argc, "-seed")) seed = strtoul(cmdOptionParser(argv+1, argv+argc, "-o"), &endptr, 10);
        
		
		if (cmdOptionExist(argv+1, argv+argc, "--help")) {if (rank == 0) print_usage();};

    }
	else {
		if (rank == 0) print_usage();	
		return -1;
	}
	
	return 0;
	
}

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
	
	int N_RA; // number of HVC(RA) neurons
	int N_I = 0; // number of HVC(I) neurons
	
    std::string outputDirectory; // directory to which write network information
	
	double Gee_max; // max strength of HVC-RA -> HVC-RA connection
	double Gie_max = 0.0;  // max strength of HVC-I -> HVC-RA connection
	double Gei_max = 0.0;  // max strength of HVC-RA -> HVC-I connection
	
	double pie = 0.0; // probability to contact an HVC-RA neuron by interneuron
	double pei = 0.0; // probability to contact an HVC-I neuron by HVC-RA neuron
	
	double mean_delay = 0.0; // mean axonal conductance delay for HVC-RA -> HVC-RA
	double sd_delay = 0.0; // standard deviation of axonal conductance delay for HVC-RA -> HVC-RA
	double synchronous_window = 1.0;  // synchronous window for inputs arrival, which defines a polychronous network
	
	int num_output; // number of output connections (same as number of neurons in each layer for synfire chain)
	int max_num_inputs; // max number of input connections for HVC-RA
	
	int num_parallel_chains = 1; // number of parallel synfire chains
	
	bool synfire = false; // indicator for sampling one or more synfire chains not embedded into space
	bool polychronous = false; // indicator for sampling polychronous network not embedded into space
	bool integration_times_uniform = false; // indicator for sampling polychronous network with a delta delay distribution and 
									// uniformly distributed neuronal integration times
	
	bool integration_times_lognormal = false; // indicator for sampling polychronous network with a delta delay distribution and 
									// truncated lognormally distributed neuronal integration times
	
	std::pair<double,double> int_range; // range of integration times
	double int_mean; // mean integration time
	double int_std; // std integration times
	
	unsigned seed = 1991; // seed for random number generators
	
	
	if (parse_command_line(argc, argv, rank, synfire, polychronous, integration_times_uniform, integration_times_lognormal, 
						outputDirectory, N_RA, N_I, seed, num_output, max_num_inputs, num_parallel_chains,
						Gee_max, Gei_max, Gie_max, pei, pie, synchronous_window, mean_delay, sd_delay, int_range, int_mean, int_std) < 0){
		MPI_Finalize();
		return 0;
	}
	
	if (rank == 0){
		std::cout << "Number of HVC-RA neurons: N_RA = " << N_RA << std::endl;
		std::cout << "Number of output connections for HVC-RA: num_output = " << num_output << std::endl;
		std::cout << "Max strength of HVC-RA -> HVC-RA connection: Gee_max = " << Gee_max << std::endl;	
		std::cout << "Number of HVC-I neurons: N_I = " << N_I << std::endl;
		std::cout << "Path to output directory: " << outputDirectory << std::endl;
		std::cout << "Seed for random number generators = " << seed << std::endl;		
		
		if (N_I > 0){
			std::cout << "Max strength of HVC-RA -> HVC-I  connection: Gei_max = " << Gei_max << std::endl;
			std::cout << "Max strength of HVC-I  -> HVC-RA connection: Gie_max = " << Gie_max << std::endl;
			std::cout << "Probability for an HVC-RA neuron to connect HVC-I:  pei = " << pei << std::endl;
			std::cout << "Probability for an HVC-I  neuron to connect HVC-RA: pie = " << pie << std::endl;
		}
			
		
	}

	HvcNetwork hvc(seed);
	
	hvc.set_connection_strengths(Gee_max, Gei_max, Gie_max);
	hvc.sample_network_without_RA2RA(N_RA, N_I, pei, pie, mean_delay, sd_delay);
		
	if (synfire){ // sample synfire chain
		if (rank == 0){
			write_training_neurons(num_output * num_parallel_chains, outputDirectory);
		
			std::cout << "\nSampling synfire chain\n"
					  << "Number of parallel chains: " << num_parallel_chains << "\n"
					  << "Mean axonal conduction delay: " << mean_delay << "\n"
					  << "Std axonal conduction delay: " << sd_delay << std::endl;
		}
		
		hvc.wire_parallel_chains_from_network_without_RA2RA_connections(num_parallel_chains, num_output, 
																			mean_delay, sd_delay, outputDirectory);
	}
		
																		
		
	if (polychronous){
		if (rank == 0){
			write_training_neurons(num_output, outputDirectory);
			std::cout << "\nSampling polychronous network\n"
					  << "Synchronous window: " << synchronous_window << " ms\n"
					  << "Mean axonal conduction delay: " << mean_delay << " ms\n"
					  << "Std axonal conduction delay: " << sd_delay << " ms" << std::endl;
		}
		
		hvc.prepare_network_for_polychronous_wiring(outputDirectory + "training_neurons.bin");
		hvc.wire_polychronous_network_customDelays(num_output, max_num_inputs, synchronous_window, mean_delay, sd_delay, outputDirectory);
	}
	
	if (integration_times_uniform){
		if (rank == 0){
			write_training_neurons(num_output, outputDirectory);
			std::cout << "\nSampling polychronous network with uniform integration times\n"
					  << "Synchronous window: " << synchronous_window << " ms\n"
					  << "Min integration time: " << int_range.first << " ms\n"
					  << "Max integration time: " << int_range.second << " ms" << std::endl;
		}
		
		hvc.prepare_network_for_polychronous_wiring(outputDirectory + "training_neurons.bin");
		hvc.wire_polychronous_network_integrationTimes_uniform(num_output, max_num_inputs, synchronous_window, int_range, outputDirectory);	
	}
	
	if (integration_times_lognormal){
		if (rank == 0){
			write_training_neurons(num_output, outputDirectory);
			std::cout << "\nSampling polychronous network with truncated lognormal integration times\n"
					  << "Synchronous window: " << synchronous_window << " ms\n"
					  << "Min integration time: " << int_range.first << " ms\n"
					  << "Max integration time: " << int_range.second << " ms\n"
					  << "Mean integration time: " << int_mean << " ms\n"
					  << "Std integration time: " << int_std << " ms" << std::endl;
		}
		
		hvc.prepare_network_for_polychronous_wiring(outputDirectory + "training_neurons.bin");
		hvc.wire_polychronous_network_integrationTimes_lognormal(num_output, max_num_inputs, synchronous_window, int_range, int_mean, int_std, outputDirectory);	
	}
	
	MPI_Finalize();
	
	return 0;

}

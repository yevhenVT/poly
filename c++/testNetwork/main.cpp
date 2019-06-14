#include "HvcNetwork.h"
#include "utils.h"
#include <mpi.h>
#include <string>
#include <fstream>

using namespace std;

static void print_usage(){
	std::cout << "Usage: Simulation of network dynamics requires the following parameters to be specified:\n"
			  << "-o <output directory>\n"
			  << "-n <directory with network>\n"
			  << "-f <prefix of output filenames>"
			  << "-nt <number of simulation trials>\n"
			  << "-d <duration of a single simulation trial>\n"
			  << "-s <time spread of training neurons>\n\n"
			 
			  << "Synaptic strenghts of connections between neurons can be resampled by provided one of the following parameters:\n"
			  << "--resamle to resample connections between HVC-RA neurons\n"
			  << "--resample-all to resample connections between HVC-RA neurons and between HVC-RA and HVC-I\n\n"
			  
			  << "When --resample is specified, the following parameter is required:\n"
			  << "-gee <new max synaptic strength for HVC-RA -> HVC-RA connections>\n\n"
			
			  << "When --resample-all is specified, the following parameters are required:\n"
			  << "-gee <new max synaptic strength for HVC-RA -> HVC-RA connections>\n"
			  << "-gei <new max synaptic strength for HVC-RA -> HVC-I connections>\n"
			  << "-gie <new max synaptic strength for HVC-I  -> HVC-RA connections>\n\n"
			  
			  << "User can provide a seed for random number generators:\n" 
			  << "-seed <non-negative number>\n\n"
			  
			  << "Membrane potentials of 100 random HVC-RA neurons will be recorded if the following parameter is specified:\n"
			  << "--record\n"
			  
			  << "\nTo see the usage message type --help:\n\n"
			  << std::endl;	
}

static int parse_command_line(int argc, char** argv, int rank, bool& resample, bool& resample_all, bool& record,
								std::string& netDir, std::string& outDir, std::string& filePrefix, 
								double& gee, double& gei, double& gie, int& nt, unsigned& seed, double& duration, double& spread){
	if (argc > 1){
        if (cmdOptionExist(argv+1, argv+argc, "--resample")) resample = true;
        if (cmdOptionExist(argv+1, argv+argc, "--resample-all")) resample_all = true;
        if (cmdOptionExist(argv+1, argv+argc, "--record")) record = true;
        
        if ( (resample) && (resample_all) ){
			if (rank == 0) {std::cout << "Only one parameter --resample or --resample-all can be provided!\n" <<std::endl; print_usage();}	
			return -1;	
		}
        
		// read common information:
		// network directory; output directory; 
		// number of simulation trials; duration of a single trial
		if (cmdOptionExist(argv+1, argv+argc, "-n")) netDir = cmdOptionParser(argv+1, argv+argc, "-n");
        else {
			if (rank == 0) {std::cout << "Network directory -n is missing!\n" <<std::endl; print_usage();}	
			return -1;	
		}
		
		if (cmdOptionExist(argv+1, argv+argc, "-o")) outDir = cmdOptionParser(argv+1, argv+argc, "-o");
        else {
			if (rank == 0) {std::cout << "Output directory -o is missing!\n" <<std::endl; print_usage();}	
			return -1;	
		}
		
		if (cmdOptionExist(argv+1, argv+argc, "-f")) filePrefix = cmdOptionParser(argv+1, argv+argc, "-f");
        else {
			if (rank == 0) {std::cout << "Prefix for output filenames -f is missing!\n" <<std::endl; print_usage();}	
			return -1;	
		}
		
        if (cmdOptionExist(argv+1, argv+argc, "-nt")) nt = atoi(cmdOptionParser(argv+1, argv+argc, "-nt"));
        else {
			if (rank == 0) {std::cout << "Number of simulation trials is missing!\n" <<std::endl; print_usage();}
			return -1;	
		}
		
		if (cmdOptionExist(argv+1, argv+argc, "-d")) duration = atoi(cmdOptionParser(argv+1, argv+argc, "-d"));
        else {
			if (rank == 0){std::cout << "Trial duration -d is missing!\n" <<std::endl; print_usage();}
			return -1;	
		}
		
		if (cmdOptionExist(argv+1, argv+argc, "-s")) spread = atoi(cmdOptionParser(argv+1, argv+argc, "-s"));
        else {
			if (rank == 0){std::cout << "Spread of training neurons -s is missing!\n" <<std::endl; print_usage();}
			return -1;	
		}
		
		// if resample indicator is on, new synaptic strength of HVC-RA connection has to be provided
		if ( resample ){		
			if (cmdOptionExist(argv+1, argv+argc, "-gee")) gee = atof(cmdOptionParser(argv+1, argv+argc, "-gee"));
			else {
				if (rank == 0){std::cout << "Strength of HVC-RA -> HVC-RA connection -gee is missing!\n" <<std::endl; print_usage();}
				return -1;	
			}
		}
		
		// if resample_all indicator is on, new synaptic strengths for all connections have to be provided
		if ( resample_all ){		
			if (cmdOptionExist(argv+1, argv+argc, "-gee")) gee = atof(cmdOptionParser(argv+1, argv+argc, "-gee"));
			else {
				if (rank == 0){std::cout << "Strength of HVC-RA -> HVC-RA connection -gee is missing!\n" <<std::endl; print_usage();}
				return -1;	
			}	
			if (cmdOptionExist(argv+1, argv+argc, "-gie")) gie = atof(cmdOptionParser(argv+1, argv+argc, "-gie"));
			else {
				if (rank == 0){std::cout << "Strength of HVC-I -> HVC-RA connection -gie is missing!\n" <<std::endl; print_usage();}
				return -1;	
			}
		
			if (cmdOptionExist(argv+1, argv+argc, "-gei")) gei = atof(cmdOptionParser(argv+1, argv+argc, "-gei"));
			else {
				if (rank == 0){std::cout << "Strength of HVC-RA -> HVC-I connection -gei is missing!\n" <<std::endl; print_usage();}
				return -1;	
			}
		}
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

int main(int argc, char** argv)
{
	std::string networkDir; // directory with network
	std::string outputDir; // output directory with simulation results
	std::string filePrefix; // prefix for output filenames
	
	double Gie_max = 0.0; // max strength of HVC-I -> HVC-RA connections
	double Gee_max; // max strength of HVC-RA -> HVC-RA connections
	double Gei_max = 0.0; // max strength of HVC-RA -> HVC-I connections
	bool resample = false; // indicator if strength of HVC-RA -> HVC-RA connections are resampled
	bool resample_all = false; // indicator if strengths of all connections are resampled
	bool record = false; // indicator if membrane potentials of HVC-RA neurons are recorded
	int num_trials; // number of simulation trials
	double trial_duration; // trial duration in ms
	double training_spread; // time spread of training (starter) neurons
	
	unsigned seed = 1991; // seed for random number generators
	
    int rank; // MPI process rank
    

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    
    if (parse_command_line(argc, argv, rank, resample, resample_all, record,
								networkDir, outputDir, filePrefix,
								Gee_max, Gei_max, Gie_max, num_trials, seed, trial_duration, training_spread) < 0){
		MPI_Finalize();
		return 0;
	}
								
	if (rank == 0)
	{
		std::cout << "Path to directory with HVC network: " << networkDir << std::endl;
		std::cout << "Path to output directory: " << outputDir << std::endl;
		std::cout << "Prefix for the output filenames: " << filePrefix << std::endl;
		
		std::cout << "num_trials = " << num_trials << std::endl;
		std::cout << "trial_duration = " << trial_duration << " ms" << std::endl;		
		std::cout << "training_spread = " << training_spread << " ms" << std::endl;		
		std::cout << "Seed for random number generators = " << seed << std::endl;		
	}

	HvcNetwork hvc(seed);
	
	if ( (resample) || (resample_all) )
		hvc.set_connection_strengths(Gee_max, Gei_max, Gie_max);
	
	if ( (resample) && (rank == 0) )
		std::cout << "Resampling strengths of HVC-RA -> HVC-RA connections\n"
				  << "new Gee_max = " << Gee_max << std::endl;

	if ( (resample_all) && (rank == 0) ){
		std::cout << "Resampling strengths of all connections\n"
				  << "new Gee_max = " << Gee_max << "\n"
				  << "new Gei_max = " << Gei_max << "\n"
				  << "new Gie_max = " << Gie_max << std::endl;
		
		resample = resample_all;
	}
	
	std::string fileTraining = networkDir + "training_neurons.bin";

	hvc.prepare_network_for_testing(networkDir, fileTraining, resample);
	hvc.read_capacitance((networkDir + "cm_dend_and_integration_times.bin").c_str(), (outputDir + "noise.bin").c_str());
	double scale = 1.0;
	hvc.scale_weights(scale);
	hvc.test_network(num_trials, trial_duration, training_spread, record, outputDir, filePrefix);
	
	MPI_Finalize();


	return 0;

}


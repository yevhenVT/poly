#ifndef HVC_NETWORK_H
#define HVC_NETWORK_H

#include <vector>
#include "poisson_noise.h"
#include <cmath>
#include <string>
#include <functional>
#include <mpi.h>
#include <set>
#include <map>

#include "HH2.h"
#include "HHI.h"

class HvcNetwork
{
public:
    HvcNetwork(unsigned seed);
        
    // set functions
    void set_connection_strengths(double gee_max, double gei_max, double gie_max); // set conductance strength for all connections
	
	void wire_random_feedforward(int num_output, double mean_delay, double sd_delay,
											std::string outputDir); // wire a random feedforward network. Neurons
																	// are ordered according to their putative burst onset order.
																	// Then num_output connections are send for each neuron to the
																	// neurons that are supposed to spike later. 
	
	void wire_polychronous_network_integrationTimes_lognormal(int num_outputs, int max_num_inputs,
			double synch_margin, std::pair<double,double> int_range, double int_mean, double int_std, std::string outputDir); // // wire a polychronous network with a delta delay distribution and a truncated lognormal integration times distribution
														 
	
	void wire_polychronous_network_integrationTimes_uniform(int num_outputs, int max_num_inputs,
			double synch_margin, std::pair<double,double> int_range, std::string outputDir); // wire a polychronous network with a delta delay distribution and a uniform distribution of integration times
														 
	
	void wire_polychronous_network_customDelays(int num_outputs, int max_num_inputs,
			double synch_margin, double mean_delay, double sd_delay, std::string outputDir); // wire causal network
																							 // with delay distribution sampled from lognormal distribution with defined mean and standard deviation
																							 // and with defined synchronization window of inputs
	
	void wire_parallel_chains_from_network_without_RA2RA_connections(int num_chains, int num_neurons_in_layer, 
																		double mean_delay, double sd_delay,
																		std::string outputDir); // sample several parallel chains

	void rewire_fraction_connections(double fraction, std::string netDir, std::string outputDir); // rewire fraction of HVC-RA -> HVC-RA synapses to random targets
	
	void sample_network_without_RA2RA(int N_ra, int N_i, double pei, double pie, double mean_delay, double sd_delay); // sample HVC network without HVC-RA -> HVC-RA connections
	
	// prepare networks
	void prepare_network_for_polychronous_wiring(const std::string& fileTraining); // prepare HVC network for sampling causal network:
											
	void prepare_network_for_testing(std::string networkDir, std::string fileTraining, bool resample); // read network configuration, resample synaptic strengths if needed and distribute among slaves
		
	void test_network(int num_trials, double trial_duration, double spread, bool record, std::string outputDirectory, std::string fileSimName); // run multiple simulation of network dynamics
	
	void read_capacitance(const char* file_capcitance, const char* file_noise); // read dendritic capacitance from file_capacitance and write neuronal noise to file_noise	
private:
	/////////////////
	// HVC parameters
	/////////////////
	
	// number of neurons in network
	int N_TR; // number of training HVC(RA) neurons
	int N_RA; // number of HVC(RA) neurons
	int N_I; // number of HVC(I) neurons
	
	
	// connectivity params
	double Gie_max; // max synaptic weight from HVC(I) to HVC(RA) neuron
    double Gei_max; // max strength of HVC(RA) -> HVC(I) connection
	double Gee_max; // max synaptic weight from HVC(RA) to HVC(RA) neuron
    
    //////////////////////////////////////////
    // HVC parameters for dynamics simulations
    //////////////////////////////////////////
    std::vector<int> training_neurons; // array with indices of HVC(RA) training neurons
	
    // neurons
	vector<HH2> HVCRA_local; // array of HVC(RA) neurons
	vector<HHI> HVCI_local; // array of HVC(I) neurons
		
	////////////////////////////////////////////////////	
	// MPI support and work distribution among processes
	////////////////////////////////////////////////////
	
	int MPI_size; // number of MPI processes
	int MPI_rank; // rank of MPI process

	int N_RA_local; // number of HVC(RA) neurons to handle in process for sampling connections and time delays
	int N_I_local;	// // number of HVC(I) neurons to handle in process for sampling connections and time delays

	std::vector<int> Id_RA_local; // array with ids of HVC(RA) neurons to handle in process for sampling connections and time delays
	std::vector<int> Id_I_local;	// array with ids of HVC(I) neurons to handle in process for sampling connections and time delays

	std::vector<int> N_RA_sizes; // array with number of RA neurons in each process for sampling connections and time delays
	std::vector<int> N_I_sizes; // array with nubmer of I neurons in each process for sampling connections and time delays
	
	Poisson_noise noise_generator; // noise generator
	
	////////////////////////////////////////////////////////////////
	// global arrays stored on master process for network generation
	////////////////////////////////////////////////////////////////
	
	// synaptic targets
	std::vector<std::vector<int>> syn_ID_RA_I; // array with synaptic ID numbers from HVC(RA) to HVC(I) neurons
	std::vector<std::vector<int>> syn_ID_I_RA; // array with synaptic ID numbers from HVC(I) to HVC(RA) neurons
	std::vector<std::vector<int>> syn_ID_RA_RA; // array with synaptic ID numbers from HVC(RA) to HVC(RA) neurons
	
	// synaptic weights
	std::vector<std::vector<double>> weights_RA_I; // array with synaptic weights from HVC(RA) to HVC(I) neurons
	std::vector<std::vector<double>> weights_I_RA; // array with synaptic weights from HVC(I) to HVC(RA) neurons
	std::vector<std::vector<double>> weights_RA_RA; // matrix with synaptic weights from HVC(RA) to HVC(RA) neurons
	
	// axonal time delays
	std::vector<std::vector<double>> axonal_delays_RA_I; // array with axonal delays for HVC(RA) -> HVC(I) connections
	std::vector<std::vector<double>> axonal_delays_RA_RA; // array with axonal delays for HVC(RA) -> HVC(RA) connections
	std::vector<std::vector<double>> axonal_delays_I_RA; // array with axonal delays for HVC(I) -> HVC(RA) connections
	
	/////////////////////////////////////////////////////////
	// local arrays stored on slaves for dynamics simulations
	/////////////////////////////////////////////////////////
	
	// synaptic targets
	std::vector<std::vector<int>> syn_ID_RA_I_local; // array with synaptic ID numbers from HVC(RA) to HVC(I) neurons
	std::vector<std::vector<int>> syn_ID_I_RA_local; // array with synaptic ID numbers from HVC(I) to HVC(RA) neurons
	std::vector<std::vector<int>> syn_ID_RA_RA_local; // array with synaptic ID numbers from HVC(RA) to HVC(RA) neurons
	
	// synaptic weights
	std::vector<std::vector<double>> weights_RA_I_local; // array with synaptic weights from HVC(RA) to HVC(I) neurons
	std::vector<std::vector<double>> weights_I_RA_local; // array with synaptic weights from HVC(I) to HVC(RA) neurons
	std::vector<std::vector<double>> weights_RA_RA_local; // matrix with synaptic weights from HVC(RA) to HVC(RA) neurons
	
	// axonal time delays
	std::vector<std::vector<double>> axonal_delays_RA_I_local; // array with axonal delays for HVC(RA) -> HVC(I) connections
	std::vector<std::vector<double>> axonal_delays_RA_RA_local; // array with axonal delays for HVC(RA) -> HVC(RA) connections
	std::vector<std::vector<double>> axonal_delays_I_RA_local; // array with axonal delays for HVC(I) -> HVC(RA) connections
	
	
	// supporting arrays for dynamics simulations
	std::vector<std::vector<double>> spikes_in_trial_soma_global; // array with somatic spike times in the last trial
	std::vector<std::vector<double>> spikes_in_trial_dend_global; // array with dendritic spike times in the last trial
	std::vector<std::vector<double>> spikes_in_trial_interneuron_global; // array with interneuron spike times in the last trial

	std::vector<std::vector<double>> spikes_in_trial_soma_local; // array with somatic spike times in the last trial
	std::vector<std::vector<double>> spikes_in_trial_dend_local; // array with dendritic spike times in the last trial
	std::vector<std::vector<double>> spikes_in_trial_interneuron_local; // array with interneuron spike times in the last trial
	
	///////////////////////////
	// Simulation parameters //
	///////////////////////////
	const static double WAITING_TIME; // waiting time before current is injected to training neurons
	const static double TIMESTEP; // time step for dynamics
	const static double NETWORK_UPDATE_FREQUENCY; // how often network state should be updated in ms

	const static double CONDUCTANCE_PULSE; // strength of excitatory conductance pulse delivered to training neurons
	
	// noise
	const static double WHITE_NOISE_MEAN_SOMA; // dc component of white noise to soma
	const static double WHITE_NOISE_STD_SOMA; // variance of white noise to soma
	const static double WHITE_NOISE_MEAN_DEND; // dc component of white noise to dendrite
	const static double WHITE_NOISE_STD_DEND; // variance of white noise to dendrite

	    	
    // FUNCTIONS
	
	//////////
	// current
	//////////
	void set_training_conductance_pulse(); // deliver excitatory conductance pulse to the training neurons

    //////////
    // pruning
    //////////
    
	void remove_connections_to_neurons(int neuron_id); // completely remove HVC-RA -> HVC-RA connections to neuron with id neuron_id									
																										    
    // MISC
    
    //////////////
    // MPI Support
    //////////////
    void distribute_work(); // distribute neurons evenly among processes
    void prepare_slaves_for_testing(std::string fileTraining); // send synapses and delays from master to slaves;
															   // initialize network for dynamics simulations
    
    void get_neuronRA_location(int n, int* rank, int* shift); // get location of RA neuron with ID n in terms of process and position in array
	void get_neuronI_location(int n, int* rank, int* shift); // get location of I neuron with ID n in terms of process and position in array	
	
    void gather_data(); // gather data from all processes
    void gather_local_arrays(const std::vector<double> &local, std::vector<double> &global); // gather data from local arrays to global, assuming sizes of arrays 
				
	// send functions
    void scatter_connections_RA2RA(); // send HVC(RA) -> HVC(RA) connections from master process to all slaves
    void scatter_connections_betweenRAandI(); // send connections between HVC(RA) and HVC(I) from master process to all slaves

	void scatter_global_to_local_double(const std::vector<double>& global,
												std::vector<double>& local); // scatter global array from master process (size N_RA)
																			 // to local arrays on slaves (size N_RA_local each)
    ///////////////////////////
    // causal network functions
    ///////////////////////////
    int wire_polychronous_network_integrationTimes_iteration(int min_num_neurons_to_connect, double time_to_connect,
								double synch_margin, int num_outputs, std::vector<int>& num_inputs, int max_num_inputs,
								std::vector<bool>& indicators_connected_to_targets,
								std::vector<double>& assigned_burst_labels, std::vector<double>& integration_times,
								std::set<int>& network_neurons); // an iteration of the wiring of a polychronous network with zero delays
																 // and different integration times
    
    int wire_polychronous_network_customDelays_iteration(int min_num_neurons_to_connect, double time_to_connect, 
								double synch_margin, double mean_delay, double sd_delay,
								int num_outputs, std::vector<int>& num_inputs, int max_num_inputs,
								std::vector<bool>& indicators_connected_to_targets,
								std::vector<double>& assigned_burst_labels,
								std::set<int>& network_neurons);
			
	////////////////////////
	// sample connections //
	////////////////////////
    void resample_strength_of_connections_in_network(); // resample synaptic strengths based on existing synaptic connections.
														// Synapses must be already loaded by function read_experimental_network_from_directory
	
    std::vector<double> sample_axonal_delays_from_lognormal(int N, double mean, double var); // sample N delays from lognormal distribution with mean and variance
    
	double sample_Ge2i(); // sample synaptic weight from HVC(RA) to HVC(I) neuron
    double sample_Gi2e(); // sample synaptic weight from HVC(I) to HVC(RA) neuron
    double sample_Ge2e(); // sample synaptic weight from HVC(RA) to HVC(RA) neuron
    
	void sample_capacitance_and_integration_times_uniform(int N, std::pair<double,double> int_range, 
						std::vector<double>& capacitance_dend, std::vector<double>& integration_times); // sample dendritic capacitance from a uniform distribution and as a result integration times for neurons
    
    void sample_capacitance_and_integration_times_lognormal(int N, std::pair<double,double> int_range, double int_mean, double int_std,
						std::vector<double>& capacitance_dend, std::vector<double>& integration_times); // sample dendritic capacitance from a truncated lognormal distribution with a cut-off low value.
																										// based on capacitance, sample integration times
    
    void sample_noise_based_on_dend_capacitance(const std::vector<double>& cm,
				std::vector<double>& mu_soma, std::vector<double>& std_soma, 
				std::vector<double>& mu_dend, std::vector<double>& std_dend); // sample neuronal noise strength based on their dendritic capacitance
    /////////////////
    // initialization
    /////////////////
    
    void initialize_generators(unsigned seed); // initialize generator for processes
    
    void initialize_dynamics(); // prepare arrays and neurons for simulating dynamics
	
	void resize_arrays_for_I(int n_local, int n_total); // resize data arrays for HVC(I) neurons
    void resize_arrays_for_RA(int n_local, int n_total); // resize data arrays for HVC(RA) neurons
    void resize_global_arrays(); // resize data arrays for HVC network
					 
   	////////////////
   	// set functions
   	////////////////
   	void set_noise(); // set noise for all neurons in the network. White noise is used for HVC(RA) and poisson noise for HVC(I)
	void set_dynamics(); // initialize vector arrays of proper size for dynamics of HVC(RA) and HVC(I) neurons
    
    void set_noise_parameters(const struct NoiseParameters& noise_params); // set noise parameters
    void set_time_parameters(const struct TimeParameters& time_params); // set time parameters
    
    void set_noise_based_on_dend_capacitance(const std::vector<double>& cm); // set noise level based on dendritic capacitance
    ///////////
    // dynamics
    ///////////																		
    std::vector<double> trial_distributed_training(double trial_duration, double spread, bool record_excitatory_current); // simulation trial of mature network with training neurons having uniform random spread
																				// excitatory current of the neurons can be tracked and returned
    
	void run_polychronous_network(double trial_duration); // simulation trial for causal wiring. Simulating stops when simulation time exceeds stop_time    
    
    void randomize_after_trial(); // set all neurons to the resting state
        
   		
   
  
    // read from files    
													   
	void read_synapses(std::vector<std::vector<int>>& syn_ID,
					   std::vector<std::vector<double>>& weights,
					   std::vector<std::vector<double>>& axonal_delays,
					   const char* filename); // read synapses from a file
	
    void read_training_neurons(const char* filename); // read training neurons from file    
    void read_number_of_neurons(const char* filename); // read number of neurons in the network
																	  
   
    
    // write to files
    void write_noise_based_on_dend_capacitance(const std::vector<double>& mu_soma, const std::vector<double>& std_soma, 
													   const std::vector<double>& mu_dend, const std::vector<double>& std_dend,
													   const char* filename); // write noise of neurons which was set based on their dendritic capacitance
    
    void write_capacitance_and_integration_time(const std::vector<double>& c, 
										const std::vector<double>& it, const char* filename); // write capacitance and integration times of neurons to a file
										
    void write_training_neurons(const char* filename); // write training neurons to a file
    
    void write_burst_labels(const std::vector<double>& labels, const char* filename); // write burst labels to file
    
    void write_synapses(const std::vector<std::vector<int>>& syn_ID,
						const std::vector<std::vector<double>>& weights,
						const std::vector<std::vector<double>>& axonal_delays,
						const char* filename); // write synapses to a file
        
	void write_parameters_to_file(const char* outputDirectory); // write parameters of synaptic strengths and neuron numbers to file	

	void write_soma_spike_times(const char* filename); // write somatic spike information to a file
    void write_dend_spike_times(const char* filename); // write dendritic spike information to a file
     
    void write_interneuron_spike_times(const char* filename); // write interneuron spike information to a file
	void write_array(std::vector<double> a, const char* filename); // write array to a file


	void write_network_test(int num_trials, std::vector<int>& total_num_dend_spikes, std::vector<double>& average_num_dendritic_spikes_in_trials, 
                              std::vector<double>& average_num_somatic_spikes_in_trials, std::vector<double>& mean_burst_time, 
                              std::vector<double>& std_burst_time, const char* filename); // write results of network test to file
                              
    void write_experimental_network_to_directory(std::string outputDirectory); // experimental network to output directory

        
};
	
	
template<typename T>
std::vector<T> sample_randomly_noreplacement_from_array(std::vector<T> a, int sample_size, Poisson_noise &noise_generator)
{
	if (sample_size > static_cast<int>(a.size()))
	{
		std::cout << "Sample size exceeds the array size!\n" << std::endl;
		return std::vector<T>();
	}
	
	std::vector<T> sample(sample_size);
	
	for (int i = 0; i < sample_size; i++)
	{
		int sample_ind = noise_generator.sample_integer(0, a.size()-1-i);
		
		sample[i] = a[sample_ind];
		a[sample_ind] = a[a.size()-1-i];
		
	}
	return sample;
}

#endif

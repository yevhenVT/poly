#include "HvcNetwork.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>
#include <set>
#include <tuple>
#include <unordered_map>
#include <cassert>
#include <sstream>
#include <unordered_set>
#include <functional>

using namespace std::placeholders;

const double HvcNetwork::WAITING_TIME = 100.0; // waiting time in ms before current injection to training neurons
const double HvcNetwork::CONDUCTANCE_PULSE = 3.0; // conductance pulse delivered to training neurons

const double HvcNetwork::TIMESTEP = 0.01; // time step for dynamics in ms
const double HvcNetwork::NETWORK_UPDATE_FREQUENCY = 0.1; // how often network state should be updated in ms

const double HvcNetwork::WHITE_NOISE_MEAN_SOMA = 0.0; // dc component of white noise to soma
const double HvcNetwork::WHITE_NOISE_STD_SOMA = 0.0; // variance of white noise to soma; default 0.1
const double HvcNetwork::WHITE_NOISE_MEAN_DEND = 0.0; // dc component of white noise to dendrite
const double HvcNetwork::WHITE_NOISE_STD_DEND = 0.0; // variance of white noise to dendrite; default 0.2

			
HvcNetwork::HvcNetwork(unsigned seed)
{   
	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
    
    this->initialize_generators(seed);
	
    Gie_max = -1.0;
    Gei_max = -1.0; // max strength of HVC(RA) -> HVC(I) connectio
    Gee_max = -1.0;
}

void HvcNetwork::wire_random_feedforward(int num_output, double mean_delay, double sd_delay,
											std::string outputDir){
	N_TR = num_output;
	
	if (MPI_rank == 0){
		std::vector<int> possible_targets(N_RA-1);
		
		std::iota(possible_targets.begin(), possible_targets.end(), 1);
		std::reverse(possible_targets.begin(), possible_targets.end());
		
		for (int i = 0; i < N_RA-num_output; i++){
			std::vector<int> targets = sample_randomly_noreplacement_from_array(possible_targets, num_output, noise_generator);
			std::vector<double> delays(num_output);
					
			if (mean_delay > 1e-3)
				delays = sample_axonal_delays_from_lognormal(num_output, mean_delay, sd_delay);
					
			
			for (int j = 0; j < num_output; j++){
				syn_ID_RA_RA[i].push_back(targets[j]);
				axonal_delays_RA_RA[i].push_back(delays[j]);
				weights_RA_RA[i].push_back(this->sample_Ge2e());
			}
			
			possible_targets.pop_back();		
		}								
		
		for (int i = N_RA-num_output; i < N_RA-1; i++){
			std::vector<double> delays(N_RA-i-1);
					
			if (mean_delay > 1e-3)
				delays = sample_axonal_delays_from_lognormal(N_RA-i-1, mean_delay, sd_delay);
			
			//std::cout << "delays.size = " << delays.size() << std::endl;
			for (int j = i+1; j < N_RA; j++){
				syn_ID_RA_RA[i].push_back(j);
				axonal_delays_RA_RA[i].push_back(delays[j-i-1]);
				weights_RA_RA[i].push_back(this->sample_Ge2e());
			}	
		}
		this->write_experimental_network_to_directory(outputDir);		
	}	
}	

void HvcNetwork::wire_parallel_chains_from_network_without_RA2RA_connections(int num_chains, int num_neurons_in_layer, 
																		double mean_delay, double sd_delay,
																		std::string outputDir)
{
	N_TR = num_chains * num_neurons_in_layer;
	
	if (MPI_rank == 0)
	{
		// estimate number of layers
		int num_layers = N_RA / (num_chains * num_neurons_in_layer);
		int num_neurons_in_chain = num_layers * num_neurons_in_layer; // number of neurons in each parallel chain
		
		// separate neurons in individual pools
		std::vector<std::vector<std::vector<int>>> parallel_chains(num_chains);	
		
		for (int i = 0; i < num_chains; i++)
		{
			parallel_chains[i].resize(num_layers);
			
			for (int j = 0; j < num_layers; j++)
				parallel_chains[i][j].resize(num_neurons_in_layer);
		}
		
		// populate chains
		for (int j = 0; j < num_layers; j++)
			for (int i = 0; i < num_chains; i++)
				std::iota(parallel_chains[i][j].begin(), parallel_chains[i][j].end(), j * num_chains * num_neurons_in_layer + i * num_neurons_in_layer);
		
		
		////////////////////////////////////////////////////////
		// Version with one connection between a pair of neurons
		////////////////////////////////////////////////////////
		
		// make all-to-all connections from neurons in one chain layer to the neurons in the next layer
		for (int i = 0; i < num_chains; i++)
		{
			for (int j = 1; j < num_layers; j++)
			{
				std::vector<double> delays(num_neurons_in_layer);
				
				if (mean_delay > 1e-3)
					delays = sample_axonal_delays_from_lognormal(num_neurons_in_layer, mean_delay, sd_delay);
					
				for (int k = 0; k < num_neurons_in_layer; k++)
				{
					int source_id = parallel_chains[i][j-1][k];
					
					for (int n = 0; n < num_neurons_in_layer; n++)
					{
						int target_id = parallel_chains[i][j][n];
						
						weights_RA_RA[source_id].push_back(this->sample_Ge2e());
						syn_ID_RA_RA[source_id].push_back(target_id);
						axonal_delays_RA_RA[source_id].push_back(delays[k]);
					}
				}
			}
		}
		
		// write results to a directory
		this->write_experimental_network_to_directory(outputDir);		
	}
}

void HvcNetwork::scatter_global_to_local_double(const std::vector<double>& global,
												std::vector<double>& local){
	
	int *sendcounts = new int[MPI_size];
    int *displs = new int[MPI_size];
	
	sendcounts[0] = N_RA_sizes[0];
	displs[0] = 0;

	for (int i = 1; i < MPI_size; i++)
	{
		sendcounts[i] = N_RA_sizes[i];
		displs[i] = displs[i-1] + sendcounts[i-1];
	}
	
	local.resize(N_RA_local);
	
	// send number of targets to all processes
	MPI_Scatterv(&global[0], sendcounts, displs, MPI_DOUBLE, 
					&local[0], N_RA_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	delete[] sendcounts;
	delete[] displs;
}

void HvcNetwork::wire_polychronous_network_integrationTimes_uniform(int num_outputs, int max_num_inputs,
			double synch_margin, std::pair<double, double> int_range, std::string outputDir)
{
	std::set<int> network_neurons; // connected HVC-RA neurons that are supposed to burst
	std::vector<double> assigned_burst_labels; // array which stores assigned burst labels for added neurons
	std::vector<double> integration_times; // array which stores integration times of neurons in the network
	std::vector<double> capacitance_dend_local; // local array which stores integration times of neurons in the network
	std::vector<double> capacitance_dend_global; // global array which stores integration times of neurons in the network
	
	std::vector<bool> indicators_neuron_connected; // indicator that output connections were sampled for this HVC-RA
	std::vector<int> num_inputs; // number of input connections for neurons
		
	if (MPI_rank == 0)
	{
		assigned_burst_labels.resize(N_RA);
		
		std::fill(assigned_burst_labels.begin(), assigned_burst_labels.end(), -1.0);		

		
		this->sample_capacitance_and_integration_times_uniform(N_RA, int_range, capacitance_dend_global, integration_times);
		
		// keep dendritic capacitance of training neurons at mean value
		// to achieve synchronous spiking of training neurons	
		double mean_capacitance = std::accumulate(capacitance_dend_global.begin(), capacitance_dend_global.end(), 0.0) / static_cast<double>(N_RA);
		double mean_integration_time = std::accumulate(integration_times.begin(), integration_times.end(), 0.0) / static_cast<double>(N_RA);
		
		for (size_t i = 0; i < training_neurons.size(); i++){
			network_neurons.insert(training_neurons[i]);
			capacitance_dend_global[training_neurons[i]] = mean_capacitance;
			integration_times[training_neurons[i]] = mean_integration_time;
		}
			
		indicators_neuron_connected.resize(N_RA);
		num_inputs.resize(N_RA);
		
		std::fill(indicators_neuron_connected.begin(), indicators_neuron_connected.end(), false);
		std::fill(num_inputs.begin(), num_inputs.end(), 0);	
		//this->write_training_neurons((outputDir + "training_neurons.bin").c_str());
		//std::cout << "Global array\n"
		//		  << "Training neurons:\n";
		//for (int i = 0; i < N_RA_sizes[0] + N_RA_sizes[1]; i++){
		//	if (i == N_RA_sizes[0]) std::cout << "Rank 1\n";
		//	std::cout << capacitance_dend_global[i] << ", " << integration_times[i] << std::endl;
		//}
		
		this->write_capacitance_and_integration_time(capacitance_dend_global, 
										integration_times, (outputDir + "cm_dend_and_integration_times.bin").c_str());
	
	}
	
	
	// send dendritic capacitance to slaves
	this->scatter_global_to_local_double(capacitance_dend_global, capacitance_dend_local);
	
	
	//if (MPI_rank == 1) std::cout << "Local array\n";
		
	// set dendritic capacitance for neurons
	for (int i = 0; i < N_RA_local; i++){
		HVCRA_local[i].set_cm_dend(capacitance_dend_local[i]);
		//if (MPI_rank == 1)
		//	std::cout << capacitance_dend_local[i] << std::endl;
	}
	
	int iter = 1;
	
	double trial_extend = 20.0; // in ms; trial duration is extended by trial_extend value
	double stop_time; // trial duration
	double max_burst_time; // maximum burst time of HVC-RA neuron in the network
	
	int min_num_neurons_to_connect = 1;
	double time_to_connect = 2.0;
	
	int continue_growth = 1;
	int save_iter = 1;
	
	// make first iteration to get spike timing of training neurons
	stop_time = WAITING_TIME + trial_extend;
	
	this->run_polychronous_network(stop_time);
	
	// change burst labels of training neurons
	if (MPI_rank == 0){
		for (int i = 0; i < N_TR; i++)
			if ( !spikes_in_trial_soma_global[training_neurons[i]].empty() )
				assigned_burst_labels[training_neurons[i]] = spikes_in_trial_soma_global[training_neurons[i]][0];
	
		max_burst_time = *std::max_element(assigned_burst_labels.begin(), assigned_burst_labels.end());
	}
	
	this->randomize_after_trial();
	
	MPI_Bcast(&max_burst_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	stop_time = max_burst_time + trial_extend;
	
	while (continue_growth == 1)
	{
		if (MPI_rank == 0)
			std::cout << "Iteration " << iter << "\n" << std::endl;
			
		this->run_polychronous_network(stop_time);
		
		if (MPI_rank == 0)
		{	
			//~ // print burst labels
			//~ for (int j = 0; j < N_RA; j++)
			//~ {
				//~ if ( spikes_in_trial_soma_global[j].size() > 0 )
					//~ std::cout << "Neuron " << j << " burst time = " << spikes_in_trial_soma_global[j][0] << " burst label = " << assigned_burst_labels[j] << "\n";
					//~ 
			//~ }
			
					
			std::cout << std::endl;
			
											  
			if (this->wire_polychronous_network_integrationTimes_iteration(min_num_neurons_to_connect, time_to_connect,
								synch_margin, num_outputs, num_inputs, max_num_inputs,
								indicators_neuron_connected,
								assigned_burst_labels, integration_times,
								network_neurons) < 0 )
				continue_growth = 0;
			
			
			if ( network_neurons.size() / 1000 == save_iter )
			{			
				std::string fileSimName = "e" + std::to_string(Gee_max) + "_i" + std::to_string(Gie_max) + "_";
			
				this->write_dend_spike_times((outputDir + fileSimName + "dendSpikes.bin").c_str());
				this->write_soma_spike_times((outputDir + fileSimName + "somaSpikes.bin").c_str());
			
				this->write_interneuron_spike_times((outputDir + fileSimName + "interneuron_spikes.bin").c_str());
				
				this->write_experimental_network_to_directory(outputDir);
				
				std::string fileBurstLabels = outputDir + "burstLabels.bin";
	
				this->write_burst_labels(assigned_burst_labels, fileBurstLabels.c_str());
				save_iter += 1;
				
			}
			
			///////////////////////////////////////
			// trial duration for sampled labels //
			///////////////////////////////////////
			max_burst_time = *std::max_element(assigned_burst_labels.begin(), assigned_burst_labels.end());
			
			
			
			std::cout << "Connected network size = " << network_neurons.size() << "\n" << std::endl;
			std::cout << "Max burst time = " << max_burst_time << "\n" << std::endl;
			
			if (network_neurons.size() >= N_RA)
				continue_growth = 0;
		}
		
		this->randomize_after_trial();
		
		// send new connections and axonal delays to slaves
		this->scatter_connections_RA2RA();
		
		MPI_Bcast(&continue_growth, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&max_burst_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		stop_time = max_burst_time + trial_extend;

		
		iter += 1;
	}
	
	stop_time = max_burst_time + trial_extend;

	std::string fileSimName = "e" + std::to_string(Gee_max) + "_i" + std::to_string(Gie_max) + "_";

		
	this->run_polychronous_network(stop_time);
		
	if (MPI_rank == 0) 
	{
		//~ std::cout << "Network neurons:\n";
		//~ 
		//~ for (auto it = network_neurons.begin(); it != network_neurons.end(); it++)
			//~ std::cout << *it << ", ";
		//~ std::cout << "\n" << std::endl;
		//~ 
		// write burst labels
		std::string fileBurstLabels = outputDir + "burstLabels.bin";
	
		this->write_burst_labels(assigned_burst_labels, fileBurstLabels.c_str());
		this->write_dend_spike_times((outputDir + fileSimName + "dendSpikes.bin").c_str());
		this->write_soma_spike_times((outputDir + fileSimName + "somaSpikes.bin").c_str());
			
		this->write_interneuron_spike_times((outputDir + fileSimName + "interneuron_spikes.bin").c_str());
	}
}

void HvcNetwork::wire_polychronous_network_integrationTimes_lognormal(int num_outputs, int max_num_inputs,
			double synch_margin, std::pair<double,double> int_range, double int_mean, double int_std, std::string outputDir)
{
	std::set<int> network_neurons; // connected HVC-RA neurons that are supposed to burst
	std::vector<double> assigned_burst_labels; // array which stores assigned burst labels for added neurons
	std::vector<double> integration_times; // array which stores integration times of neurons in the network
	std::vector<double> capacitance_dend_local; // local array which stores integration times of neurons in the network
	std::vector<double> capacitance_dend_global; // global array which stores integration times of neurons in the network
	
	std::vector<bool> indicators_neuron_connected; // indicator that output connections were sampled for this HVC-RA
	std::vector<int> num_inputs; // number of input connections for neurons
		
	if (MPI_rank == 0)
	{
		assigned_burst_labels.resize(N_RA);
		
		std::fill(assigned_burst_labels.begin(), assigned_burst_labels.end(), -1.0);		

		
		this->sample_capacitance_and_integration_times_lognormal(N_RA, int_range, int_mean, int_std, capacitance_dend_global, integration_times);
		
		// keep dendritic capacitance of training neurons at mean value
		// to achieve synchronous spiking of training neurons	
		double mean_capacitance = std::accumulate(capacitance_dend_global.begin(), capacitance_dend_global.end(), 0.0) / static_cast<double>(N_RA);
		double mean_integration_time = std::accumulate(integration_times.begin(), integration_times.end(), 0.0) / static_cast<double>(N_RA);
		
		for (size_t i = 0; i < training_neurons.size(); i++){
			network_neurons.insert(training_neurons[i]);
			capacitance_dend_global[training_neurons[i]] = mean_capacitance;
			integration_times[training_neurons[i]] = mean_integration_time;
		}
			
		indicators_neuron_connected.resize(N_RA);
		num_inputs.resize(N_RA);
		
		std::fill(indicators_neuron_connected.begin(), indicators_neuron_connected.end(), false);
		std::fill(num_inputs.begin(), num_inputs.end(), 0);	
		//this->write_training_neurons((outputDir + "training_neurons.bin").c_str());
		//std::cout << "Global array\n"
		//		  << "Training neurons:\n";
		//for (int i = 0; i < N_RA_sizes[0] + N_RA_sizes[1]; i++){
		//	if (i == N_RA_sizes[0]) std::cout << "Rank 1\n";
		//	std::cout << capacitance_dend_global[i] << ", " << integration_times[i] << std::endl;
		//}
		
		this->write_capacitance_and_integration_time(capacitance_dend_global, 
										integration_times, (outputDir + "cm_dend_and_integration_times.bin").c_str());
	
	}
	
	// send dendritic capacitance to slaves
	this->scatter_global_to_local_double(capacitance_dend_global, capacitance_dend_local);
	
	//if (MPI_rank == 1) std::cout << "Local array\n";
		
	// set dendritic capacitance for neurons
	for (int i = 0; i < N_RA_local; i++){
		HVCRA_local[i].set_cm_dend(capacitance_dend_local[i]);
		//if (MPI_rank == 1)
		//	std::cout << capacitance_dend_local[i] << std::endl;
	}
	
	int iter = 1;
	
	double trial_extend = 20.0; // in ms; trial duration is extended by trial_extend value
	double stop_time; // trial duration
	double max_burst_time; // maximum burst time of HVC-RA neuron in the network
	
	int min_num_neurons_to_connect = 1;
	double time_to_connect = 2.0;
	
	int continue_growth = 1;
	int save_iter = 1;
	
	// make first iteration to get spike timing of training neurons
	stop_time = WAITING_TIME + trial_extend;
	
	this->run_polychronous_network(stop_time);
	
	// change burst labels of training neurons
	if (MPI_rank == 0){
		for (int i = 0; i < N_TR; i++)
			if ( !spikes_in_trial_soma_global[training_neurons[i]].empty() )
				assigned_burst_labels[training_neurons[i]] = spikes_in_trial_soma_global[training_neurons[i]][0];
	
		max_burst_time = *std::max_element(assigned_burst_labels.begin(), assigned_burst_labels.end());
	}
	
	this->randomize_after_trial();
	
	MPI_Bcast(&max_burst_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	stop_time = max_burst_time + trial_extend;
	
	while (continue_growth == 1)
	{
		if (MPI_rank == 0)
			std::cout << "Iteration " << iter << "\n" << std::endl;
			
		this->run_polychronous_network(stop_time);
		
		if (MPI_rank == 0)
		{	
			// print burst labels
			for (int j = 0; j < N_RA; j++)
			{
				if ( spikes_in_trial_soma_global[j].size() > 0 )
					std::cout << "Neuron " << j << " burst time = " << spikes_in_trial_soma_global[j][0] << " burst label = " << assigned_burst_labels[j] << "\n";
					
			}
			
					
			std::cout << std::endl;
			
											  
			if (this->wire_polychronous_network_integrationTimes_iteration(min_num_neurons_to_connect, time_to_connect,
								synch_margin, num_outputs, num_inputs, max_num_inputs,
								indicators_neuron_connected,
								assigned_burst_labels, integration_times,
								network_neurons) < 0 )
				continue_growth = 0;
			
			
			
			
			///////////////////////////////////////
			// trial duration for sampled labels //
			///////////////////////////////////////
			max_burst_time = *std::max_element(assigned_burst_labels.begin(), assigned_burst_labels.end());
			
			
			
			std::cout << "Connected network size = " << network_neurons.size() << "\n" << std::endl;
			std::cout << "Max burst time = " << max_burst_time << "\n" << std::endl;
			
			if (network_neurons.size() >= N_RA)
				continue_growth = 0;
				
			if (( network_neurons.size() / 1000 == save_iter ) || (continue_growth == 0))
			{		
				std::string fileSimName = "e" + std::to_string(Gee_max) + "_i" + std::to_string(Gie_max) + "_";
			
				this->write_dend_spike_times((outputDir + fileSimName + "dendSpikes.bin").c_str());
				this->write_soma_spike_times((outputDir + fileSimName + "somaSpikes.bin").c_str());
			
				this->write_interneuron_spike_times((outputDir + fileSimName + "interneuron_spikes.bin").c_str());
				
				this->write_experimental_network_to_directory(outputDir);
				
				std::string fileBurstLabels = outputDir + "burstLabels.bin";
	
				this->write_burst_labels(assigned_burst_labels, fileBurstLabels.c_str());
				save_iter += 1;
				
			}
		}
		
		this->randomize_after_trial();
		
		// send new connections and axonal delays to slaves
		this->scatter_connections_RA2RA();
		
		MPI_Bcast(&continue_growth, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&max_burst_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		stop_time = max_burst_time + trial_extend;

		
		iter += 1;
	}
	
	stop_time = max_burst_time + trial_extend;

	std::string fileSimName = "e" + std::to_string(Gee_max) + "_i" + std::to_string(Gie_max) + "_";

		
	this->run_polychronous_network(stop_time);
		
	if (MPI_rank == 0) 
	{
		//~ std::cout << "Network neurons:\n";
		//~ 
		//~ for (auto it = network_neurons.begin(); it != network_neurons.end(); it++)
			//~ std::cout << *it << ", ";
		//~ std::cout << "\n" << std::endl;
		//~ 
		// write burst labels
		std::string fileBurstLabels = outputDir + "burstLabels.bin";
	
		this->write_burst_labels(assigned_burst_labels, fileBurstLabels.c_str());
		this->write_dend_spike_times((outputDir + fileSimName + "dendSpikes.bin").c_str());
		this->write_soma_spike_times((outputDir + fileSimName + "somaSpikes.bin").c_str());
			
		this->write_interneuron_spike_times((outputDir + fileSimName + "interneuron_spikes.bin").c_str());
	}
}

void HvcNetwork::rewire_fraction_connections(double fraction, std::string networkDir, std::string outputDir){
	if (MPI_rank == 0)
	{
		std::string fileParameters = networkDir + "parameters.bin";
	
		this->read_number_of_neurons(fileParameters.c_str()); 
		
		std::string filename_connections_RA2I = networkDir + "RA_I_connections.bin";
		std::string filename_connections_I2RA = networkDir + "I_RA_connections.bin";
		std::string filename_connections_RA2RA = networkDir + "RA_RA_connections.bin";

		this->read_synapses(syn_ID_RA_RA, weights_RA_RA, axonal_delays_RA_RA, filename_connections_RA2RA.c_str());
		this->read_synapses(syn_ID_RA_I,  weights_RA_I,  axonal_delays_RA_I,  filename_connections_RA2I.c_str());
		this->read_synapses(syn_ID_I_RA,  weights_I_RA,  axonal_delays_I_RA,  filename_connections_I2RA.c_str());
		
		for (int i = 0; i < N_RA; i++){
			if (!syn_ID_RA_RA[i].empty()){
				int num_connections_to_rewire = static_cast<int>(fraction*syn_ID_RA_RA[i].size());
				std::vector<int> target_indices(syn_ID_RA_RA[i].size());
				std::iota(target_indices.begin(), target_indices.end(), 0);
				
				std::vector<int> target_indices_to_rewire = sample_randomly_noreplacement_from_array(target_indices, num_connections_to_rewire, noise_generator);
				
				//std::cout << "Neuron " << i << " with " << syn_ID_RA_RA[i].size() 
				//		  << " outputs and " << num_connections_to_rewire << " connections to rewire\n";
				
				//for (size_t j = 0; j < syn_ID_RA_RA[i].size(); j++)
				//	std::cout << syn_ID_RA_RA[i][j] << " ";
				//std::cout << "\n" << std::endl;
				
				
				//for (size_t j = 0; j < target_indices_to_rewire.size(); j++)
				//	std::cout << target_indices_to_rewire[j] << "," << syn_ID_RA_RA[i][target_indices_to_rewire[j]] << " ";
				//std::cout << "\n" << std::endl;
				
				std::vector<int> possible_new_targets(N_RA-1-static_cast<int>(syn_ID_RA_RA[i].size()));
				std::set<int> old_targets_set(syn_ID_RA_RA[i].begin(), syn_ID_RA_RA[i].end());
				
				int possible_target_counter = 0;
				for (int j = 0; j < N_RA; j++)
					if ((j!=i)&&(old_targets_set.find(j)==old_targets_set.end())){
						possible_new_targets[possible_target_counter] = j;
						possible_target_counter += 1;
					}
				
				std::vector<int> new_targets = sample_randomly_noreplacement_from_array(possible_new_targets, num_connections_to_rewire, noise_generator);
				
				//std::cout << "New targets\n";
				//for (size_t j = 0; j < new_targets.size(); j++)
				//	std::cout << new_targets[j] << " ";
				//std::cout << "\n" << std::endl;
				
				
				for (int j = 0; j < num_connections_to_rewire; j++)
					syn_ID_RA_RA[i][target_indices_to_rewire[j]] = new_targets[j];	
			}
		}
		this->write_experimental_network_to_directory(outputDir);
	}	
	
}

void HvcNetwork::wire_polychronous_network_customDelays(int num_outputs, int max_num_inputs,
			double synch_margin, double mean_delay, double sd_delay, std::string outputDir)
{
	std::set<int> network_neurons; // connected HVC-RA neurons that are supposed to burst
	std::vector<double> assigned_burst_labels; // array which stores assigned burst labels for added neurons
	std::vector<bool> indicators_neuron_connected; // indicator that output connections were sampled for this HVC-RA
	std::vector<int> num_inputs; // number of input connections for neurons
		
	if (MPI_rank == 0)
	{
		assigned_burst_labels.resize(N_RA);
		
		std::fill(assigned_burst_labels.begin(), assigned_burst_labels.end(), -1.0);		

			
		for (size_t i = 0; i < training_neurons.size(); i++)
			network_neurons.insert(training_neurons[i]);
			
		
		
		indicators_neuron_connected.resize(N_RA);
		num_inputs.resize(N_RA);
		
		std::fill(indicators_neuron_connected.begin(), indicators_neuron_connected.end(), false);
		std::fill(num_inputs.begin(), num_inputs.end(), 0);	
		
		//this->write_training_neurons((outputDir + "training_neurons.bin").c_str());
	}
		
	int iter = 1;
	
	double trial_extend = 20.0; // in ms; trial duration is extended by trial_extend value
	double stop_time; // trial duration
	double max_burst_time; // maximum burst time of HVC-RA neuron in the network
	
	int min_num_neurons_to_connect = 1;
	double time_to_connect = 2.0;
	
	int continue_growth = 1;
	int save_iter = 1;
	
	// make first iteration to get spike timing of training neurons
	stop_time = WAITING_TIME + trial_extend;
	
	this->run_polychronous_network(stop_time);
	
	// change burst labels of training neurons
	if (MPI_rank == 0){
		for (int i = 0; i < N_TR; i++)
			if ( !spikes_in_trial_soma_global[training_neurons[i]].empty() )
				assigned_burst_labels[training_neurons[i]] = spikes_in_trial_soma_global[training_neurons[i]][0];
	
		max_burst_time = *std::max_element(assigned_burst_labels.begin(), assigned_burst_labels.end());
	}
	
	this->randomize_after_trial();
	
	MPI_Bcast(&max_burst_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	stop_time = max_burst_time + trial_extend;
	
	while (continue_growth == 1)
	{
		if (MPI_rank == 0)
			std::cout << "Iteration " << iter << "\n" << std::endl;
			
		this->run_polychronous_network(stop_time);
		
		if (MPI_rank == 0)
		{	
			//~ // print burst labels
			//~ for (int j = 0; j < N_RA; j++)
			//~ {
				//~ if ( spikes_in_trial_soma_global[j].size() > 0 )
					//~ std::cout << "Neuron " << j << " burst time = " << spikes_in_trial_soma_global[j][0] << " burst label = " << assigned_burst_labels[j] << "\n";
					//~ 
			//~ }
			
					
			std::cout << std::endl;
			
											  
			if ( this->wire_polychronous_network_customDelays_iteration(min_num_neurons_to_connect, time_to_connect, 
												  synch_margin, mean_delay, sd_delay, num_outputs,
												  num_inputs, max_num_inputs, indicators_neuron_connected,
												  assigned_burst_labels, network_neurons) < 0 )
				continue_growth = 0;
			
			
			if ( network_neurons.size() / 1000 == save_iter )
			{			
				std::string fileSimName = "e" + std::to_string(Gee_max) + "_i" + std::to_string(Gie_max) + "_";
			
				this->write_dend_spike_times((outputDir + fileSimName + "dendSpikes.bin").c_str());
				this->write_soma_spike_times((outputDir + fileSimName + "somaSpikes.bin").c_str());
			
				this->write_interneuron_spike_times((outputDir + fileSimName + "interneuron_spikes.bin").c_str());
				
				this->write_experimental_network_to_directory(outputDir);
				
				std::string fileBurstLabels = outputDir + "burstLabels.bin";
	
				this->write_burst_labels(assigned_burst_labels, fileBurstLabels.c_str());
				save_iter += 1;
				
			}
			
			///////////////////////////////////////
			// trial duration for sampled labels //
			///////////////////////////////////////
			max_burst_time = *std::max_element(assigned_burst_labels.begin(), assigned_burst_labels.end());
			
			
			
			std::cout << "Connected network size = " << network_neurons.size() << "\n" << std::endl;
			std::cout << "Max burst time = " << max_burst_time << "\n" << std::endl;
			
			if (network_neurons.size() >= N_RA)
				continue_growth = 0;
		}
		
		this->randomize_after_trial();
		
		// send new connections and axonal delays to slaves
		this->scatter_connections_RA2RA();
		
		MPI_Bcast(&continue_growth, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&max_burst_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		stop_time = max_burst_time + trial_extend;

		
		iter += 1;
	}
	
	stop_time = max_burst_time + trial_extend;

	std::string fileSimName = "e" + std::to_string(Gee_max) + "_i" + std::to_string(Gie_max) + "_";

		
	this->run_polychronous_network(stop_time);
		
	if (MPI_rank == 0) 
	{
		//~ std::cout << "Network neurons:\n";
		//~ 
		//~ for (auto it = network_neurons.begin(); it != network_neurons.end(); it++)
			//~ std::cout << *it << ", ";
		//~ std::cout << "\n" << std::endl;
		//~ 
		// write burst labels
		std::string fileBurstLabels = outputDir + "burstLabels.bin";
	
		this->write_burst_labels(assigned_burst_labels, fileBurstLabels.c_str());
		this->write_dend_spike_times((outputDir + fileSimName + "dendSpikes.bin").c_str());
		this->write_soma_spike_times((outputDir + fileSimName + "somaSpikes.bin").c_str());
			
		this->write_interneuron_spike_times((outputDir + fileSimName + "interneuron_spikes.bin").c_str());
	}
}

int HvcNetwork::wire_polychronous_network_integrationTimes_iteration(int min_num_neurons_to_connect, double time_to_connect,
								double synch_margin, int num_outputs, std::vector<int>& num_inputs, int max_num_inputs,
								std::vector<bool>& indicators_connected_to_targets,
								std::vector<double>& assigned_burst_labels, std::vector<double>& integration_times,
								std::set<int>& network_neurons)
{		
	// wire connections so that they arrive synchronously
	
	// create sorted array with burst times of all neurons
	std::vector<std::pair<double, int>> all_first_spike_times_sorted; // sorted first spike times of neurons
	std::vector<std::pair<double,int>> burst_labels_of_silent_sorted; // sorted assigned burst labels of silent neurons


	for (int i = 0; i < N_RA; i++)
	{
		int num_bursts = static_cast<int>(spikes_in_trial_dend_global[i].size());
		
		//if ( num_bursts > 1 )
		//	std::cout << "Neuron " << i << " bursts " << num_bursts << " times!\n" << std::endl;
		
		if ( (num_bursts == 0) && ( assigned_burst_labels[i] > 0 ) )
			burst_labels_of_silent_sorted.push_back(std::pair<double,int>(assigned_burst_labels[i], i));
			
		
		if ( !spikes_in_trial_soma_global[i].empty() )
			all_first_spike_times_sorted.push_back(std::pair<double,int>(spikes_in_trial_soma_global[i][0], i));
	}
	
	
	// sort burst times and labels
	std::sort(all_first_spike_times_sorted.begin(), all_first_spike_times_sorted.end());
	std::sort(burst_labels_of_silent_sorted.begin(), burst_labels_of_silent_sorted.end()); 
	
	
	// generate array with candidates to connect	
	std::multimap<double, int> candidate_targets; // list of labeled candidates for targeting: (burst time, neuron_id) 
	
	std::set<int> new_source_neurons;
	
	std::vector<std::pair<double,int>> new_source_first_spike_times_sorted;
	
	double max_wired_burst_time = -100;
	
	// find the the largest burst time of wired neuron
	for (size_t i = 0; i < all_first_spike_times_sorted.size(); i++)
	{
		int neuron_id = all_first_spike_times_sorted[i].second;
		double burst_time = all_first_spike_times_sorted[i].first;
		
		if (( indicators_connected_to_targets[neuron_id] ) && (burst_time > max_wired_burst_time) )
			max_wired_burst_time = burst_time;
		
	}
	
	double earliest_new_source_burst = max_wired_burst_time - 10.0;
	
	// sample new source neurons
	
	for (size_t i = 0; i < all_first_spike_times_sorted.size(); i++)
	{
		int neuron_id = all_first_spike_times_sorted[i].second;
		
		if ( ( !indicators_connected_to_targets[neuron_id] ) && (all_first_spike_times_sorted[i].first > earliest_new_source_burst) )
		{
			if ( !new_source_first_spike_times_sorted.empty() )
			{
				if ( ( new_source_first_spike_times_sorted.size() < min_num_neurons_to_connect ) || ( all_first_spike_times_sorted[i].first - new_source_first_spike_times_sorted[0].first <= time_to_connect) )
				{
					new_source_first_spike_times_sorted.push_back(all_first_spike_times_sorted[i]);
					new_source_neurons.insert(neuron_id);
				}
				
				else if ( ( new_source_first_spike_times_sorted.size() >= min_num_neurons_to_connect ) && ( all_first_spike_times_sorted[i].first - new_source_first_spike_times_sorted[0].first > time_to_connect ) )
					break;	
					
			}
			
			else
			{
				new_source_first_spike_times_sorted.push_back(all_first_spike_times_sorted[i]);
				new_source_neurons.insert(neuron_id);
			}				
		}
	}
	
	if (new_source_neurons.empty())	{
		std::cout << "No new source neurons!\n" << std::endl;
		return -1;
	}
	
	static int num_removed_silent_neurons = 0;
	
	//Don't keep non-robust neurons	
	// find silent neurons with burst labels smaller than smallest burst time of source neuron
	for (size_t i = 0; i < burst_labels_of_silent_sorted.size(); i++)
	{
		if ( burst_labels_of_silent_sorted[i].first < new_source_first_spike_times_sorted[0].first )
		{
			int neuron_id = burst_labels_of_silent_sorted[i].second;
			
			std::cout << "Burst label of silent neuron: " << burst_labels_of_silent_sorted[i].first
			          << " num_inputs = " << num_inputs[neuron_id] << " Smallest first spike time of source: " << new_source_first_spike_times_sorted[0].first
			          << " Largest first spike time of source: " << new_source_first_spike_times_sorted.back().first << "\n" << std::endl;
		
			// remove silent neurons from network
			// erase all synapses to neuron
			this->remove_connections_to_neurons(neuron_id);
			num_removed_silent_neurons++;
			
			// update number of inputs for connected neurons
			for (size_t j = 0; j < syn_ID_RA_RA[neuron_id].size(); j++)
			{
				int target_id = syn_ID_RA_RA[neuron_id][j];
				
				num_inputs[target_id] -= 1;
			}
			
			
			// erase all synapses from neuron
			std::vector<int>().swap(syn_ID_RA_RA[neuron_id]);
			std::vector<double>().swap(axonal_delays_RA_RA[neuron_id]);
			std::vector<double>().swap(weights_RA_RA[neuron_id]);			
			
			assigned_burst_labels[neuron_id] = -1.0;
			//assigned_total_conductances[neuron_id] = 0.0;
			num_inputs[neuron_id] = 0;
			
			indicators_connected_to_targets[neuron_id] = false;
			
			auto it = network_neurons.find(neuron_id);
			
			if (it != network_neurons.end())
			{
				network_neurons.erase(it);
				//std::cout << "Neuron " << neuron_id << " was removed from network because it did not spike robustly\n" << std::endl;
			}
			else
				std::cerr << "Neuron " << neuron_id << " to remove is not found in network_neurons!\n" << std::endl;
		
		}
	}
	
	//std::cout << "Total number of removed silent neurons: " << num_removed_silent_neurons << std::endl;

	std::vector<int> num_labels_in_windows; // number of burst labels in windows
	std::vector<int> num_onsets_in_windows; // number of burst onsets in windows
	
	std::vector<std::vector<double>> burst_labels_in_windows; // burst labels of neurons in windows
	std::vector<std::vector<double>> burst_onsets_in_windows; // burst onsets of neurons in windows
	
	
	std::vector<int> num_labels_in_small_windows; // number of burst labels in windows
	std::vector<std::vector<double>> burst_labels_in_small_windows; // burst labels of neurons in small windows
	
	
	double small_window = 1.0; // was 1.0
	double window = 1.0; // was 1.0
	double min_label = 1e6;
	double min_burst_onset = all_first_spike_times_sorted[0].first;
	
	
	for (size_t i = 0; i < assigned_burst_labels.size(); i++)
	{
		if ( ( assigned_burst_labels[i] > 0) && ( assigned_burst_labels[i] < min_label ) )
			min_label = assigned_burst_labels[i];
	}
	
	std::cout << "New source neurons and their first spike times\n";
	for (size_t i = 0; i < new_source_first_spike_times_sorted.size(); i++)
		std::cout << "Source neuron " << new_source_first_spike_times_sorted[i].second 
				  << " with first spike time = " << new_source_first_spike_times_sorted[i].first << std::endl;
	
	
	
	// add candidates if 
	// - burst label is assigned;
	// - neuron is not connected to targets
	// - neuron is not among new source neurons
	// - neuron fires slightly before or later than the earliest new source neuron
	
	if ( !new_source_first_spike_times_sorted.empty() )
	{
		for (size_t i = 0; i < assigned_burst_labels.size(); i++)
		{
			if ( assigned_burst_labels[i] > 0 )
			{	
				//////////////////////////////////////
				// allow to target training neurons //
				//////////////////////////////////////
				if ( ( new_source_neurons.find(i) == new_source_neurons.end() ) && ( !indicators_connected_to_targets[i] ) 
								&& ( num_inputs[i] < max_num_inputs ) && 
								(new_source_first_spike_times_sorted[0].first - assigned_burst_labels[i] <= 5.0) )
				candidate_targets.insert(std::pair<double,int>(assigned_burst_labels[i], i));
			}
		}
	}
	else
	{
		std::cout << "No source neurons!\n" << std::endl;
		return -1;
	}
	
	
	
	// sort candidates by number of input connections
	std::multimap<int, std::pair<double, int>> candidates_by_input;
	
	for (std::multimap<double,int>::iterator it = candidate_targets.begin(); it != candidate_targets.end(); it++)
	{
		double target_label = (*it).first;
		int target_id = (*it).second;
			
		candidates_by_input.insert(std::make_pair(num_inputs[target_id], *it));	
	}
	
	
	//~ std::cout << "\nCandidate neurons for connection and their assigned burst labels\n";
	//~ for (auto it = candidate_targets.begin(); it != candidate_targets.end(); it++)
		//~ std::cout << "Candidate neuron " << (*it).second 
				  //~ << " with assigned burst label = " << (*it).first 
				  //~ << " and integration time = " << integration_times[(*it).second] << std::endl;
	
	
		
	double DELAY_EARLY_MARGIN = synch_margin / 2.0;
	double DELAY_LATE_MARGIN = synch_margin / 2.0;
	double DELAY = 3.4; // single delay value was 3.4
	int NUM_NEURONS_TO_SAMPLE = 1;
	
	double FRACTION_LOST = 0.0; // fraction of allowed lost synapses
	
	int num_new_source = static_cast<int>(new_source_neurons.size()); // number of new source neurons
	
	int total_num_outputs = num_new_source * num_outputs;
	int allowed_num_lost = static_cast<int>(static_cast<double>(total_num_outputs) * FRACTION_LOST);
	
	std::vector<double> synapse_delays(total_num_outputs);
	std::fill(synapse_delays.begin(), synapse_delays.end(), DELAY);
	
	//std::cout << "Number of sampled synapses: " << synapse_delays.size() << "\n";
	//std::cout << "Allowed number of lost synapses: " << allowed_num_lost << "\n";
	
	
	/////////////////////////////////////////////////////////////////
	// Add all source neurons									   //
	/////////////////////////////////////////////////////////////////
	
	std::vector<std::pair<double,int>> source_with_spikes = new_source_first_spike_times_sorted;
	std::set<int> source_neurons = new_source_neurons;
	
	for (size_t i = 0; i < all_first_spike_times_sorted.size(); i++)
	{
		int neuron_id = all_first_spike_times_sorted[i].second;
		
		if (indicators_connected_to_targets[neuron_id])
		{
			source_with_spikes.push_back(all_first_spike_times_sorted[i]);
			source_neurons.insert(neuron_id);
		}
		
	}
	
	std::sort(source_with_spikes.begin(), source_with_spikes.end());
	
	//std::cout << "\nAll source neurons and their first spike times\n";
	//for (size_t i = 0; i < source_with_spikes.size(); i++)
	//	std::cout << "Source neuron " << source_with_spikes[i].second 
	//			  << " with first spike time = " << source_with_spikes[i].first << std::endl;
	
	
	std::random_device rd;
    std::mt19937 rng(rd());
	
	std::vector<double> unwired_delays = synapse_delays;
	std::vector<std::tuple<int, int, double, double>> synapse_parameters; // synapses parameters in form (source_id, target_id, delay, length, weight, synapse_loc_x, synapse_loc_y, synapse_loc_z)
	
	int num_targets_sampled = 0;
	int rewire_iteration = 1;
		
	// sample randomly synapses to try fo wire them starting from the target with the smallest
	// number of inputs. If synapse cannot be wired, sample a new target.
	// Stop if number of unwired synapses is small enough
	while (true)
	{
		//std::cout << "Rewiring iteration: " << rewire_iteration << std::endl;
		//std::cout << "Num targets sampled: " << num_targets_sampled << std::endl;
		synapse_parameters.clear();
		
		std::vector<int> new_num_inputs = num_inputs; // copy number of inputs
		std::vector<std::set<int>> added_connections(N_RA);
		
		
		while ( !candidates_by_input.empty() )
		{
			auto it_target = candidates_by_input.begin();
			
			int target_id = (*it_target).second.second;
			double target_label = (*it_target).second.first;
			
			//~ std::cout << "Target with smallest number of inputs, num_inputs, target_label = "
					  //~ << target_id << ", " << num_inputs[target_id] << ", " << target_label << std::endl;
			//~ 
			bool target_was_connected = false; // indicator that target was connected by some source neuron
			
			// find source neurons that are able to connect to the target
			auto it_earliest_source = std::lower_bound(source_with_spikes.begin(), source_with_spikes.end(), std::pair<double, int>(target_label - integration_times[target_id] - DELAY - DELAY_EARLY_MARGIN, -1));
			auto it_latest_source = std::upper_bound(source_with_spikes.begin(), source_with_spikes.end(), std::pair<double, int>(target_label - integration_times[target_id] - DELAY + DELAY_LATE_MARGIN, 1e6));
			
			if (it_earliest_source != it_latest_source){
				// extract source neurons that are able to connect to the target
				// and choose source neurons randomly until match is found
				std::vector<std::pair<double,int>> source_on_time(it_earliest_source, it_latest_source);

				std::shuffle(source_on_time.begin(), source_on_time.end(), rng);

				for (auto it_source = source_on_time.begin(); it_source != source_on_time.end(); it_source++){
	
					int source_id = (*it_source).second;
					double source_first_spike_time = (*it_source).first;
					
					// ensure source is not connected to target. SINGLE CONNECTION!
					if ( std::find(syn_ID_RA_RA[source_id].begin(), syn_ID_RA_RA[source_id].end(), target_id) != syn_ID_RA_RA[source_id].end() )
						continue;
					
					// then check in array with added connections
					if ( added_connections[source_id].find(target_id) != added_connections[source_id].end() )
						continue;
				
					
					// if spike arrives on time
					if ((!unwired_delays.empty()) && (source_first_spike_time >= target_label - integration_times[target_id] - DELAY - DELAY_EARLY_MARGIN)
					   && (source_first_spike_time <= target_label - integration_times[target_id] - DELAY + DELAY_LATE_MARGIN)){
					
						new_num_inputs[target_id] += 1;
							
						double weight = this->sample_Ge2e();
						double delay = DELAY;
						
						// remove synapse
						unwired_delays.pop_back();
						
						synapse_parameters.push_back(std::make_tuple(source_id, target_id, delay, weight));
						
						added_connections[source_id].insert(target_id);
						
						target_was_connected = true;	
						break;
					}
					
					
				} // end loop through source neurons
			}
			// if target was connected by some source neuron, stop
			// reinsert target to a new place in the target array or
			// just delete it if number of inputs is exceeded
			candidates_by_input.erase(it_target);

			if ( target_was_connected )
			{
				if ( new_num_inputs[target_id] < max_num_inputs )
					candidates_by_input.insert(std::make_pair(new_num_inputs[target_id], std::make_pair(target_label, target_id)));	 
			}
			
		} // end loop through targets
		
		// if number of unwired synapses exceeds allowed amount, sample a new target
		int num_remaining_synapses = static_cast<int>(unwired_delays.size());
		
		if (num_remaining_synapses > allowed_num_lost)
		{
			// exit if network already contains N_RA neurons
			if (network_neurons.size() == N_RA) break;

			int num_to_sample = NUM_NEURONS_TO_SAMPLE;
			
			if ( num_remaining_synapses < NUM_NEURONS_TO_SAMPLE )
				num_to_sample = num_remaining_synapses;
				
			for (size_t i = 0; i < num_to_sample; i++)
			{
				// randomly sample source neuron
				int ind = noise_generator.sample_integer(0, new_source_first_spike_times_sorted.size()-1);
				int source_id = (*(new_source_first_spike_times_sorted.begin()+ind)).second;
			
				double source_first_spike_time = spikes_in_trial_soma_global[source_id][0];
				
				
				// randomly sample a target which is not in the network
				std::vector<int> sampling_pool(N_RA);
				std::iota(sampling_pool.begin(), sampling_pool.end(), 0);
				
				int new_target_id;
				
				while (true)
				{
					int ind = noise_generator.sample_integer(0, sampling_pool.size()-1);
					new_target_id = sampling_pool[ind];
				
					if (network_neurons.find(new_target_id) == network_neurons.end())
						break;
					
					sampling_pool[ind] = sampling_pool.back();
					sampling_pool.pop_back();	
				}
				
				double delay = DELAY;
				double label = source_first_spike_time + delay + integration_times[new_target_id];
				
				
				//std::cout << "New target id : " << new_target_id << std::endl;
				
				num_targets_sampled += 1;
				
				double weight = this->sample_Ge2e();
					
				syn_ID_RA_RA[source_id].push_back(new_target_id);
				weights_RA_RA[source_id].push_back(weight);
				axonal_delays_RA_RA[source_id].push_back(delay);
				 
				num_inputs[new_target_id] += 1;
							
				assigned_burst_labels[new_target_id] = label;
				
				candidate_targets.insert(std::pair<double,int>(label, new_target_id));												
				network_neurons.insert(new_target_id);
			
				// find synapse and delete from array with all source synapses
				synapse_delays.pop_back();
					
				// delete synapse from sample array
				unwired_delays.pop_back();	
			
				if (network_neurons.size() == N_RA) break;
			}
		}
				
		else
		{
			//std::cout << "Remaining unwired synapses: " << unwired_delays.size() << std::endl;
			break;
		}
		
		candidates_by_input.clear();
		// update candidate by inputs array
		for (std::multimap<double,int>::iterator it = candidate_targets.begin(); it != candidate_targets.end(); it++)
		{
			double target_label = (*it).first;
			int target_id = (*it).second;
				
			if ( num_inputs[target_id] < max_num_inputs )
				candidates_by_input.insert(std::make_pair(num_inputs[target_id], *it));	
			
		}
		
		//std::cout << "Remaining unwired synapses: " << unwired_delays.size() << std::endl;
		
		unwired_delays = synapse_delays;
		rewire_iteration++;
	} // end while there are synapses in the pool
	
	
	
	// update source arrays according to sampled synapses
	for (size_t i = 0; i < synapse_parameters.size(); i++)
	{
		int source_id = std::get<0>(synapse_parameters[i]);
		int target_id = std::get<1>(synapse_parameters[i]);
		
		double delay = std::get<2>(synapse_parameters[i]);
		double weight = std::get<3>(synapse_parameters[i]);
		
		
		
		//~ std::cout << source_id << " -> " << target_id 
				  //~ << " " << delay << " " << length << " " << weight
				  //~ << " " << std::get<0>(synapse_coord) << " " << std::get<1>(synapse_coord) << " " << std::get<2>(synapse_coord) << std::endl; 
		//~ 
		
		syn_ID_RA_RA[source_id].push_back(target_id);
		weights_RA_RA[source_id].push_back(weight);
		axonal_delays_RA_RA[source_id].push_back(delay);
		
		num_inputs[target_id] += 1;

	}
	
	for (auto it = new_source_first_spike_times_sorted.begin(); it < new_source_first_spike_times_sorted.end(); it++)
		indicators_connected_to_targets[(*it).second] = true;
	
	return 0;
}

/*int HvcNetwork::wire_polychronous_network_integrationTimes_iteration(int min_num_neurons_to_connect, double time_to_connect,
								double synch_margin, int num_outputs, std::vector<int>& num_inputs, int max_num_inputs,
								std::vector<bool>& indicators_connected_to_targets,
								std::vector<double>& assigned_burst_labels, std::vector<double>& integration_times,
								std::set<int>& network_neurons)
{		
	// wire connections so that they arrive synchronously
	
	// create sorted array with burst times of all neurons
	std::vector<std::pair<double, int>> all_first_spike_times_sorted; // sorted first spike times of neurons
	std::vector<std::pair<double,int>> burst_labels_of_silent_sorted; // sorted assigned burst labels of silent neurons


	for (int i = 0; i < N_RA; i++)
	{
		int num_bursts = static_cast<int>(spikes_in_trial_dend_global[i].size());
		
		//if ( num_bursts > 1 )
		//	std::cout << "Neuron " << i << " bursts " << num_bursts << " times!\n" << std::endl;
		
		if ( (num_bursts == 0) && ( assigned_burst_labels[i] > 0 ) )
			burst_labels_of_silent_sorted.push_back(std::pair<double,int>(assigned_burst_labels[i], i));
			
		
		if ( !spikes_in_trial_soma_global[i].empty() )
			all_first_spike_times_sorted.push_back(std::pair<double,int>(spikes_in_trial_soma_global[i][0], i));
	}
	
	
	// sort burst times and labels
	std::sort(all_first_spike_times_sorted.begin(), all_first_spike_times_sorted.end());
	std::sort(burst_labels_of_silent_sorted.begin(), burst_labels_of_silent_sorted.end()); 
	
	
	// generate array with candidates to connect	
	std::multimap<double, int> candidate_targets; // list of labeled candidates for targeting: (burst time, neuron_id) 
	
	std::set<int> new_source_neurons;
	
	std::vector<std::pair<double,int>> new_source_first_spike_times_sorted;
	
	double max_wired_burst_time = -100;
	
	// find the the largest burst time of wired neuron
	for (size_t i = 0; i < all_first_spike_times_sorted.size(); i++)
	{
		int neuron_id = all_first_spike_times_sorted[i].second;
		double burst_time = all_first_spike_times_sorted[i].first;
		
		if (( indicators_connected_to_targets[neuron_id] ) && (burst_time > max_wired_burst_time) )
			max_wired_burst_time = burst_time;
		
	}
	
	double earliest_new_source_burst = max_wired_burst_time - 10.0;
	
	// sample new source neurons
	
	for (size_t i = 0; i < all_first_spike_times_sorted.size(); i++)
	{
		int neuron_id = all_first_spike_times_sorted[i].second;
		
		if ( ( !indicators_connected_to_targets[neuron_id] ) && (all_first_spike_times_sorted[i].first > earliest_new_source_burst) )
		{
			if ( !new_source_first_spike_times_sorted.empty() )
			{
				if ( ( new_source_first_spike_times_sorted.size() < min_num_neurons_to_connect ) || ( all_first_spike_times_sorted[i].first - new_source_first_spike_times_sorted[0].first <= time_to_connect) )
				{
					new_source_first_spike_times_sorted.push_back(all_first_spike_times_sorted[i]);
					new_source_neurons.insert(neuron_id);
				}
				
				else if ( ( new_source_first_spike_times_sorted.size() >= min_num_neurons_to_connect ) && ( all_first_spike_times_sorted[i].first - new_source_first_spike_times_sorted[0].first > time_to_connect ) )
					break;	
					
			}
			
			else
			{
				new_source_first_spike_times_sorted.push_back(all_first_spike_times_sorted[i]);
				new_source_neurons.insert(neuron_id);
			}				
		}
	}
	
	if (new_source_neurons.empty())	{
		std::cout << "No new source neurons!\n" << std::endl;
		return -1;
	}
	
	static int num_removed_silent_neurons = 0;
	
	//Don't keep non-robust neurons	
	// find silent neurons with burst labels smaller than smallest burst time of source neuron
	for (size_t i = 0; i < burst_labels_of_silent_sorted.size(); i++)
	{
		if ( burst_labels_of_silent_sorted[i].first < new_source_first_spike_times_sorted[0].first )
		{
			int neuron_id = burst_labels_of_silent_sorted[i].second;
			
			std::cout << "Burst label of silent neuron: " << burst_labels_of_silent_sorted[i].first
			          << " num_inputs = " << num_inputs[neuron_id] << " Smallest first spike time of source: " << new_source_first_spike_times_sorted[0].first
			          << " Largest first spike time of source: " << new_source_first_spike_times_sorted.back().first << "\n" << std::endl;
		
			// remove silent neurons from network
			// erase all synapses to neuron
			this->remove_connections_to_neurons(neuron_id);
			num_removed_silent_neurons++;
			
			// update number of inputs for connected neurons
			for (size_t j = 0; j < syn_ID_RA_RA[neuron_id].size(); j++)
			{
				int target_id = syn_ID_RA_RA[neuron_id][j];
				
				num_inputs[target_id] -= 1;
			}
			
			
			// erase all synapses from neuron
			std::vector<int>().swap(syn_ID_RA_RA[neuron_id]);
			std::vector<double>().swap(axonal_delays_RA_RA[neuron_id]);
			std::vector<double>().swap(weights_RA_RA[neuron_id]);			
			
			assigned_burst_labels[neuron_id] = -1.0;
			//assigned_total_conductances[neuron_id] = 0.0;
			num_inputs[neuron_id] = 0;
			
			indicators_connected_to_targets[neuron_id] = false;
			
			auto it = network_neurons.find(neuron_id);
			
			if (it != network_neurons.end())
			{
				network_neurons.erase(it);
				//std::cout << "Neuron " << neuron_id << " was removed from network because it did not spike robustly\n" << std::endl;
			}
			else
				std::cerr << "Neuron " << neuron_id << " to remove is not found in network_neurons!\n" << std::endl;
		
		}
	}
	
	//std::cout << "Total number of removed silent neurons: " << num_removed_silent_neurons << std::endl;

	std::vector<int> num_labels_in_windows; // number of burst labels in windows
	std::vector<int> num_onsets_in_windows; // number of burst onsets in windows
	
	std::vector<std::vector<double>> burst_labels_in_windows; // burst labels of neurons in windows
	std::vector<std::vector<double>> burst_onsets_in_windows; // burst onsets of neurons in windows
	
	
	std::vector<int> num_labels_in_small_windows; // number of burst labels in windows
	std::vector<std::vector<double>> burst_labels_in_small_windows; // burst labels of neurons in small windows
	
	
	double small_window = 1.0; // was 1.0
	double window = 1.0; // was 1.0
	double min_label = 1e6;
	double min_burst_onset = all_first_spike_times_sorted[0].first;
	
	
	for (size_t i = 0; i < assigned_burst_labels.size(); i++)
	{
		if ( ( assigned_burst_labels[i] > 0) && ( assigned_burst_labels[i] < min_label ) )
			min_label = assigned_burst_labels[i];
	}
	
	std::cout << "New source neurons and their first spike times\n";
	for (size_t i = 0; i < new_source_first_spike_times_sorted.size(); i++)
		std::cout << "Source neuron " << new_source_first_spike_times_sorted[i].second 
				  << " with first spike time = " << new_source_first_spike_times_sorted[i].first << std::endl;
	
	
	
	// add candidates if 
	// - burst label is assigned;
	// - neuron is not connected to targets
	// - neuron is not among new source neurons
	// - neuron fires slightly before or later than the earliest new source neuron
	
	if ( !new_source_first_spike_times_sorted.empty() )
	{
		for (size_t i = 0; i < assigned_burst_labels.size(); i++)
		{
			if ( assigned_burst_labels[i] > 0 )
			{	
				//////////////////////////////////////
				// allow to target training neurons //
				//////////////////////////////////////
				if ( ( new_source_neurons.find(i) == new_source_neurons.end() ) && ( !indicators_connected_to_targets[i] ) 
								&& ( num_inputs[i] < max_num_inputs ) && 
								(new_source_first_spike_times_sorted[0].first - assigned_burst_labels[i] <= 5.0) )
				candidate_targets.insert(std::pair<double,int>(assigned_burst_labels[i], i));
			}
		}
	}
	else
	{
		std::cout << "No source neurons!\n" << std::endl;
		return -1;
	}
	
	
	
	// sort candidates by number of input connections
	std::multimap<int, std::pair<double, int>> candidates_by_input;
	
	for (std::multimap<double,int>::iterator it = candidate_targets.begin(); it != candidate_targets.end(); it++)
	{
		double target_label = (*it).first;
		int target_id = (*it).second;
			
		candidates_by_input.insert(std::make_pair(num_inputs[target_id], *it));	
	}
	
	
	//~ std::cout << "\nCandidate neurons for connection and their assigned burst labels\n";
	//~ for (auto it = candidate_targets.begin(); it != candidate_targets.end(); it++)
		//~ std::cout << "Candidate neuron " << (*it).second 
				  //~ << " with assigned burst label = " << (*it).first 
				  //~ << " and integration time = " << integration_times[(*it).second] << std::endl;
	
	
		
	double DELAY_EARLY_MARGIN = synch_margin / 2.0;
	double DELAY_LATE_MARGIN = synch_margin / 2.0;
	int NUM_NEURONS_TO_SAMPLE = 1;
	
	double FRACTION_LOST = 0.0; // fraction of allowed lost synapses
	
	int num_new_source = static_cast<int>(new_source_neurons.size()); // number of new source neurons
	
	int total_num_outputs = num_new_source * num_outputs;
	int allowed_num_lost = static_cast<int>(static_cast<double>(total_num_outputs) * FRACTION_LOST);
	
	std::vector<double> synapse_delays(total_num_outputs);
	std::fill(synapse_delays.begin(), synapse_delays.end(), 0.0);
	
	//std::cout << "Number of sampled synapses: " << synapse_delays.size() << "\n";
	//std::cout << "Allowed number of lost synapses: " << allowed_num_lost << "\n";
	
	
	/////////////////////////////////////////////////////////////////
	// Add all source neurons that spiked before but not too early //
	/////////////////////////////////////////////////////////////////
	
	std::vector<std::pair<double,int>> source_with_spikes = new_source_first_spike_times_sorted;
	std::set<int> source_neurons = new_source_neurons;
	
	
	double MAX_DELAY = 50.0;
	
	for (size_t i = 0; i < all_first_spike_times_sorted.size(); i++)
	{
		int neuron_id = all_first_spike_times_sorted[i].second;
		
		if ( (indicators_connected_to_targets[neuron_id]) && 
			(new_source_first_spike_times_sorted[0].first - all_first_spike_times_sorted[i].first <= MAX_DELAY))
		{
			source_with_spikes.push_back(all_first_spike_times_sorted[i]);
			source_neurons.insert(neuron_id);
		}
		
	}
	
	std::sort(source_with_spikes.begin(), source_with_spikes.end());
	
	std::cout << "\nAll source neurons and their first spike times\n";
	for (size_t i = 0; i < source_with_spikes.size(); i++)
		std::cout << "Source neuron " << source_with_spikes[i].second 
				  << " with first spike time = " << source_with_spikes[i].first << std::endl;
	
	
	std::random_device rd;
    std::mt19937 rng(rd());
	
	std::vector<double> unwired_delays = synapse_delays;
	std::vector<std::tuple<int, int, double, double>> synapse_parameters; // synapses parameters in form (source_id, target_id, delay, length, weight, synapse_loc_x, synapse_loc_y, synapse_loc_z)
	
	int num_targets_sampled = 0;
	int rewire_iteration = 1;
		
	// sample randomly synapses to try fo wire them starting from the target with the smallest
	// number of inputs. If synapse cannot be wired, sample a new target.
	// Stop if number of unwired synapses is small enough
	while (true)
	{
		//std::cout << "Rewiring iteration: " << rewire_iteration << std::endl;
		//std::cout << "Num targets sampled: " << num_targets_sampled << std::endl;
		synapse_parameters.clear();
		
		std::vector<int> new_num_inputs = num_inputs; // copy number of inputs
		std::vector<std::set<int>> added_connections(N_RA);
		
		// shuffle spikes of source neurons
		std::shuffle(source_with_spikes.begin(), source_with_spikes.end(), rng);

		while ( !candidates_by_input.empty() )
		{
			auto it_target = candidates_by_input.begin();
			
			int target_id = (*it_target).second.second;
			double target_label = (*it_target).second.first;
			
			//~ std::cout << "Target with smallest number of inputs, num_inputs, target_label = "
					  //~ << target_id << ", " << num_inputs[target_id] << ", " << target_label << std::endl;
			//~ 
			bool target_was_connected = false; // indicator that target was connected by some source neuron
			
			// loop through all source neurons
			for (auto it_source = source_with_spikes.begin(); it_source != source_with_spikes.end(); it_source++)
			{	
				int source_id = (*it_source).second;
				double source_first_spike_time = (*it_source).first;
				
				// ensure source is not connected to target. SINGLE CONNECTION!
				if ( std::find(syn_ID_RA_RA[source_id].begin(), syn_ID_RA_RA[source_id].end(), target_id) != syn_ID_RA_RA[source_id].end() )
					continue;
				
				// then check in array with added connections
				if ( added_connections[source_id].find(target_id) != added_connections[source_id].end() )
					continue;
			
				
				// if spike arrives on time
				if ((!unwired_delays.empty()) && (source_first_spike_time >= target_label - integration_times[target_id] - DELAY_EARLY_MARGIN)
				   && (source_first_spike_time <= target_label - integration_times[target_id] + DELAY_LATE_MARGIN)){
				
					new_num_inputs[target_id] += 1;
						
					double weight = this->sample_Ge2e();
					double delay = 0;
					
					// remove synapse
					unwired_delays.pop_back();
					
					synapse_parameters.push_back(std::make_tuple(source_id, target_id, delay, weight));
					
					added_connections[source_id].insert(target_id);
					
					target_was_connected = true;	
					break;
				}
				
				
			} // end loop through source neurons
				
			// if target was connected by some source neuron, stop
			// reinsert target to a new place in the target array or
			// just delete it if number of inputs is exceeded
			candidates_by_input.erase(it_target);

			if ( target_was_connected )
			{
				if ( new_num_inputs[target_id] < max_num_inputs )
					candidates_by_input.insert(std::make_pair(new_num_inputs[target_id], std::make_pair(target_label, target_id)));	 
			}
			
		} // end loop through targets
		
		// if number of unwired synapses exceeds allowed amount, sample a new target
		int num_remaining_synapses = static_cast<int>(unwired_delays.size());
		
		if (num_remaining_synapses > allowed_num_lost)
		{
			// exit if network already contains N_RA neurons
			if (network_neurons.size() == N_RA) break;

			int num_to_sample = NUM_NEURONS_TO_SAMPLE;
			
			if ( num_remaining_synapses < NUM_NEURONS_TO_SAMPLE )
				num_to_sample = num_remaining_synapses;
				
			for (size_t i = 0; i < num_to_sample; i++)
			{
				// randomly sample source neuron
				int ind = noise_generator.sample_integer(0, new_source_first_spike_times_sorted.size()-1);
				int source_id = (*(new_source_first_spike_times_sorted.begin()+ind)).second;
			
				double source_first_spike_time = spikes_in_trial_soma_global[source_id][0];
				
				
				// randomly sample a target which is not in the network
				std::vector<int> sampling_pool(N_RA);
				std::iota(sampling_pool.begin(), sampling_pool.end(), 0);
				
				int new_target_id;
				
				while (true)
				{
					int ind = noise_generator.sample_integer(0, sampling_pool.size()-1);
					new_target_id = sampling_pool[ind];
				
					if (network_neurons.find(new_target_id) == network_neurons.end())
						break;
					
					sampling_pool[ind] = sampling_pool.back();
					sampling_pool.pop_back();	
				}
				
				double delay = 0;
				double label = source_first_spike_time + delay + integration_times[new_target_id];
				
				
				//std::cout << "New target id : " << new_target_id << std::endl;
				
				num_targets_sampled += 1;
				
				double weight = this->sample_Ge2e();
					
				syn_ID_RA_RA[source_id].push_back(new_target_id);
				weights_RA_RA[source_id].push_back(weight);
				axonal_delays_RA_RA[source_id].push_back(delay);
				 
				num_inputs[new_target_id] += 1;
							
				assigned_burst_labels[new_target_id] = label;
				
				candidate_targets.insert(std::pair<double,int>(label, new_target_id));												
				network_neurons.insert(new_target_id);
			
				// find synapse and delete from array with all source synapses
				synapse_delays.pop_back();
					
				// delete synapse from sample array
				unwired_delays.pop_back();	
			
				if (network_neurons.size() == N_RA) break;
			}
		}
				
		else
		{
			//std::cout << "Remaining unwired synapses: " << unwired_delays.size() << std::endl;
			break;
		}
		
		candidates_by_input.clear();
		// update candidate by inputs array
		for (std::multimap<double,int>::iterator it = candidate_targets.begin(); it != candidate_targets.end(); it++)
		{
			double target_label = (*it).first;
			int target_id = (*it).second;
				
			if ( num_inputs[target_id] < max_num_inputs )
				candidates_by_input.insert(std::make_pair(num_inputs[target_id], *it));	
			
		}
		
		//std::cout << "Remaining unwired synapses: " << unwired_delays.size() << std::endl;
		
		unwired_delays = synapse_delays;
		rewire_iteration++;
	} // end while there are synapses in the pool
	
	
	
	// update source arrays according to sampled synapses
	for (size_t i = 0; i < synapse_parameters.size(); i++)
	{
		int source_id = std::get<0>(synapse_parameters[i]);
		int target_id = std::get<1>(synapse_parameters[i]);
		
		double delay = std::get<2>(synapse_parameters[i]);
		double weight = std::get<3>(synapse_parameters[i]);
		
		
		
		//~ std::cout << source_id << " -> " << target_id 
				  //~ << " " << delay << " " << length << " " << weight
				  //~ << " " << std::get<0>(synapse_coord) << " " << std::get<1>(synapse_coord) << " " << std::get<2>(synapse_coord) << std::endl; 
		//~ 
		
		syn_ID_RA_RA[source_id].push_back(target_id);
		weights_RA_RA[source_id].push_back(weight);
		axonal_delays_RA_RA[source_id].push_back(delay);
		
		num_inputs[target_id] += 1;

	}
	
	for (auto it = new_source_first_spike_times_sorted.begin(); it < new_source_first_spike_times_sorted.end(); it++)
		indicators_connected_to_targets[(*it).second] = true;
	
	return 0;
}*/

int HvcNetwork::wire_polychronous_network_customDelays_iteration(int min_num_neurons_to_connect, double time_to_connect, 
								double synch_margin, double mean_delay, double sd_delay,
								int num_outputs, std::vector<int>& num_inputs, int max_num_inputs,
								std::vector<bool>& indicators_connected_to_targets,
								std::vector<double>& assigned_burst_labels,
								std::set<int>& network_neurons)
{		
	// wire connections so that they arrive synchronously
	
	// create sorted array with burst times of all neurons
	std::vector<std::pair<double, int>> all_first_spike_times_sorted; // sorted first spike times of neurons
	std::vector<std::pair<double,int>> burst_labels_of_silent_sorted; // sorted assigned burst labels of silent neurons


	for (int i = 0; i < N_RA; i++)
	{
		int num_bursts = static_cast<int>(spikes_in_trial_dend_global[i].size());
		
		//if ( num_bursts > 1 )
		//	std::cout << "Neuron " << i << " bursts " << num_bursts << " times!\n" << std::endl;
		
		if ( (num_bursts == 0) && ( assigned_burst_labels[i] > 0 ) )
			burst_labels_of_silent_sorted.push_back(std::pair<double,int>(assigned_burst_labels[i], i));
			
		
		if ( !spikes_in_trial_soma_global[i].empty() )
			all_first_spike_times_sorted.push_back(std::pair<double,int>(spikes_in_trial_soma_global[i][0], i));
	}
	
	
	// sort burst times and labels
	std::sort(all_first_spike_times_sorted.begin(), all_first_spike_times_sorted.end());
	std::sort(burst_labels_of_silent_sorted.begin(), burst_labels_of_silent_sorted.end()); 
	
	
	// generate array with candidates to connect	
	std::multimap<double, int> candidate_targets; // list of labeled candidates for targeting: (burst time, neuron_id) 
	
	std::set<int> new_source_neurons;
	
	std::vector<std::pair<double,int>> new_source_first_spike_times_sorted;
	
	double max_wired_burst_time = -100;
	
	// find the the largest burst time of wired neuron
	for (size_t i = 0; i < all_first_spike_times_sorted.size(); i++)
	{
		int neuron_id = all_first_spike_times_sorted[i].second;
		double burst_time = all_first_spike_times_sorted[i].first;
		
		if (( indicators_connected_to_targets[neuron_id] ) && (burst_time > max_wired_burst_time) )
			max_wired_burst_time = burst_time;
		
	}
	
	double earliest_new_source_burst = max_wired_burst_time - 10.0;
	
	// sample new source neurons
	
	for (size_t i = 0; i < all_first_spike_times_sorted.size(); i++)
	{
		int neuron_id = all_first_spike_times_sorted[i].second;
		
		if ( ( !indicators_connected_to_targets[neuron_id] ) && (all_first_spike_times_sorted[i].first > earliest_new_source_burst) )
		{
			if ( !new_source_first_spike_times_sorted.empty() )
			{
				if ( ( new_source_first_spike_times_sorted.size() < min_num_neurons_to_connect ) || ( all_first_spike_times_sorted[i].first - new_source_first_spike_times_sorted[0].first <= time_to_connect) )
				{
					new_source_first_spike_times_sorted.push_back(all_first_spike_times_sorted[i]);
					new_source_neurons.insert(neuron_id);
				}
				
				else if ( ( new_source_first_spike_times_sorted.size() >= min_num_neurons_to_connect ) && ( all_first_spike_times_sorted[i].first - new_source_first_spike_times_sorted[0].first > time_to_connect ) )
					break;	
					
			}
			
			else
			{
				new_source_first_spike_times_sorted.push_back(all_first_spike_times_sorted[i]);
				new_source_neurons.insert(neuron_id);
			}				
		}
	}
	
	if (new_source_neurons.empty())	{
		std::cout << "No new source neurons!\n" << std::endl;
		return -1;
	}
	
	static int num_removed_silent_neurons = 0;
	
	//Don't keep non-robust neurons	
	// find silent neurons with burst labels smaller than smallest burst time of source neuron
	for (size_t i = 0; i < burst_labels_of_silent_sorted.size(); i++)
	{
		if ( burst_labels_of_silent_sorted[i].first < new_source_first_spike_times_sorted[0].first )
		{
			int neuron_id = burst_labels_of_silent_sorted[i].second;
			
			//std::cout << "Burst label of silent neuron: " << burst_labels_of_silent_sorted[i].first
			//          << " num_inputs = " << num_inputs[neuron_id] << " Smallest first spike time of source: " << new_source_first_spike_times_sorted[0].first
			//          << " Largest first spike time of source: " << new_source_first_spike_times_sorted.back().first << "\n" << std::endl;
		
			// remove silent neurons from network
			// erase all synapses to neuron
			this->remove_connections_to_neurons(neuron_id);
			num_removed_silent_neurons++;
			
			// update number of inputs for connected neurons
			for (size_t j = 0; j < syn_ID_RA_RA[neuron_id].size(); j++)
			{
				int target_id = syn_ID_RA_RA[neuron_id][j];
				
				num_inputs[target_id] -= 1;
			}
			
			
			// erase all synapses from neuron
			std::vector<int>().swap(syn_ID_RA_RA[neuron_id]);
			std::vector<double>().swap(axonal_delays_RA_RA[neuron_id]);
			std::vector<double>().swap(weights_RA_RA[neuron_id]);			
			
			assigned_burst_labels[neuron_id] = -1.0;
			//assigned_total_conductances[neuron_id] = 0.0;
			num_inputs[neuron_id] = 0;
			
			indicators_connected_to_targets[neuron_id] = false;
			
			auto it = network_neurons.find(neuron_id);
			
			if (it != network_neurons.end())
			{
				network_neurons.erase(it);
				//std::cout << "Neuron " << neuron_id << " was removed from network because it did not spike robustly\n" << std::endl;
			}
			else
				std::cerr << "Neuron " << neuron_id << " to remove is not found in network_neurons!\n" << std::endl;
		
		}
	}
	
	//std::cout << "Total number of removed silent neurons: " << num_removed_silent_neurons << std::endl;

	std::vector<int> num_labels_in_windows; // number of burst labels in windows
	std::vector<int> num_onsets_in_windows; // number of burst onsets in windows
	
	std::vector<std::vector<double>> burst_labels_in_windows; // burst labels of neurons in windows
	std::vector<std::vector<double>> burst_onsets_in_windows; // burst onsets of neurons in windows
	
	
	std::vector<int> num_labels_in_small_windows; // number of burst labels in windows
	std::vector<std::vector<double>> burst_labels_in_small_windows; // burst labels of neurons in small windows
	
	
	double small_window = 1.0; // was 1.0
	double window = 1.0; // was 1.0
	double min_label = 1e6;
	double min_burst_onset = all_first_spike_times_sorted[0].first;
	
	
	for (size_t i = 0; i < assigned_burst_labels.size(); i++)
	{
		if ( ( assigned_burst_labels[i] > 0) && ( assigned_burst_labels[i] < min_label ) )
			min_label = assigned_burst_labels[i];
	}
	
	//std::cout << "New source neurons and their first spike times\n";
	//for (size_t i = 0; i < new_source_first_spike_times_sorted.size(); i++)
	//	std::cout << "Source neuron " << new_source_first_spike_times_sorted[i].second 
	//			  << " with first spike time = " << new_source_first_spike_times_sorted[i].first << std::endl;
	
	
	
	// add candidates if 
	// - burst label is assigned;
	// - neuron is not connected to targets
	// - neuron is not among new source neurons
	// - neuron fires slightly before or later than the earliest new source neuron
	
	if ( !new_source_first_spike_times_sorted.empty() )
	{
		for (size_t i = 0; i < assigned_burst_labels.size(); i++)
		{
			if ( assigned_burst_labels[i] > 0 )
			{	
				//////////////////////////////////////
				// allow to target training neurons //
				//////////////////////////////////////
				if ( ( new_source_neurons.find(i) == new_source_neurons.end() ) && ( !indicators_connected_to_targets[i] ) 
								&& ( num_inputs[i] < max_num_inputs ) && 
								(new_source_first_spike_times_sorted[0].first - assigned_burst_labels[i] <= 5.0) )
				candidate_targets.insert(std::pair<double,int>(assigned_burst_labels[i], i));
			}
		}
	}
	else
	{
		std::cout << "No source neurons!\n" << std::endl;
		return -1;
	}
	
	
	
	// sort candidates by number of input connections
	std::multimap<int, std::pair<double, int>> candidates_by_input;
	
	for (std::multimap<double,int>::iterator it = candidate_targets.begin(); it != candidate_targets.end(); it++)
	{
		double target_label = (*it).first;
		int target_id = (*it).second;
			
		candidates_by_input.insert(std::make_pair(num_inputs[target_id], *it));	
	}
	
	
	//~ std::cout << "\nCandidate neurons for connection and their assigned burst labels\n";
	//~ for (auto it = candidate_targets.begin(); it != candidate_targets.end(); it++)
		//~ std::cout << "Candidate neuron " << (*it).second 
				  //~ << " with assigned burst label = " << (*it).first << std::endl;
	//~ 
	//~ 
		
	double INTEGRATION_TIME = 5.0;
	double DELAY_EARLY_MARGIN = synch_margin / 2.0;
	double DELAY_LATE_MARGIN = synch_margin / 2.0;
	int NUM_NEURONS_TO_SAMPLE = 1;
	double MEAN_DELAY = mean_delay;
	double STD_DELAY = sd_delay;
	
	double FRACTION_LOST = 0.0; // fraction of allowed lost synapses
	
	int num_new_source = static_cast<int>(new_source_neurons.size()); // number of new source neurons
	
	int total_num_outputs = num_new_source * num_outputs;
	int allowed_num_lost = static_cast<int>(static_cast<double>(total_num_outputs) * FRACTION_LOST);
	
	std::vector<double> synapse_delays = sample_axonal_delays_from_lognormal(total_num_outputs, MEAN_DELAY, STD_DELAY);
		
	
	//std::cout << "Number of sampled synapses: " << synapse_delays.size() << "\n";
	//std::cout << "Allowed number of lost synapses: " << allowed_num_lost << "\n";
	
	
	/////////////////////////////////////////////////////////////////
	// Add all source neurons that spiked before but not too early //
	/////////////////////////////////////////////////////////////////
	
	std::vector<std::pair<double,int>> source_with_spikes = new_source_first_spike_times_sorted;
	std::set<int> source_neurons = new_source_neurons;
	
	
	double MAX_DELAY = 30.0;
	
	for (size_t i = 0; i < all_first_spike_times_sorted.size(); i++)
	{
		int neuron_id = all_first_spike_times_sorted[i].second;
		
		if ( (indicators_connected_to_targets[neuron_id]) && 
			(new_source_first_spike_times_sorted[0].first - all_first_spike_times_sorted[i].first <= MAX_DELAY))
		{
			source_with_spikes.push_back(all_first_spike_times_sorted[i]);
			source_neurons.insert(neuron_id);
		}
		
	}
	
	std::sort(source_with_spikes.begin(), source_with_spikes.end());
	
	//~ std::cout << "\nAll source neurons and their first spike times\n";
	//~ for (size_t i = 0; i < source_with_spikes.size(); i++)
		//~ std::cout << "Source neuron " << source_with_spikes[i].second 
				  //~ << " with first spike time = " << source_with_spikes[i].first << std::endl;
	//~ 
	
	std::random_device rd;
    std::mt19937 rng(rd());
	
	// !!! IMPORTANT SORTING OF SYNAPTIC DELAYS !!! 
	std::sort(synapse_delays.begin(), synapse_delays.end());
	
	std::vector<double> unwired_delays = synapse_delays;
	std::vector<std::tuple<int, int, double, double>> synapse_parameters; // synapses parameters in form (source_id, target_id, delay, length, weight, synapse_loc_x, synapse_loc_y, synapse_loc_z)
	
	int num_targets_sampled = 0;
	int rewire_iteration = 1;
		
	// sample randomly synapses to try fo wire them starting from the target with the smallest
	// number of inputs. If synapse cannot be wired, sample a new target.
	// Stop if number of unwired synapses is small enough
	while (true)
	{
		//std::cout << "Rewiring iteration: " << rewire_iteration << std::endl;
		//std::cout << "Num targets sampled: " << num_targets_sampled << std::endl;
		
		synapse_parameters.clear();
		
		std::vector<int> new_num_inputs = num_inputs; // copy number of inputs
		
		std::vector<std::set<int>> added_connections(N_RA);
		
		
		// shuffle spikes of source neurons
		std::shuffle(source_with_spikes.begin(), source_with_spikes.end(), rng);

		while ( !candidates_by_input.empty() )
		{
			auto it_target = candidates_by_input.begin();
			
			int target_id = (*it_target).second.second;
			double target_label = (*it_target).second.first;
			
			//~ std::cout << "Target with smallest number of inputs, num_inputs, target_label = "
					  //~ << target_id << ", " << num_inputs[target_id] << ", " << target_label << std::endl;
			//~ 
			bool target_was_connected = false; // indicator that target was connected by some source neuron
			
			// loop through all source neurons
			for (auto it_source = source_with_spikes.begin(); it_source != source_with_spikes.end(); it_source++)
			{	
				int source_id = (*it_source).second;
				double source_first_spike_time = (*it_source).first;
				
				// ensure source is not connected to target. SINGLE CONNECTION!
				if ( std::find(syn_ID_RA_RA[source_id].begin(), syn_ID_RA_RA[source_id].end(), target_id) != syn_ID_RA_RA[source_id].end() )
					continue;
				
				// then check in array with added connections
				if ( added_connections[source_id].find(target_id) != added_connections[source_id].end() )
					continue;
			
				
				// find synapses appropriate for wiring
				auto it_synapse_low = std::lower_bound(unwired_delays.begin(), unwired_delays.end(), target_label - INTEGRATION_TIME - DELAY_EARLY_MARGIN - source_first_spike_time);
				auto it_synapse_up  = std::upper_bound(unwired_delays.begin(), unwired_delays.end(), target_label - INTEGRATION_TIME + DELAY_LATE_MARGIN  - source_first_spike_time);

				if (it_synapse_low != it_synapse_up)
				{
					std::vector<double>::iterator it_best_synch_synapse;
					double best_synch = 1e6;
					
					for (auto it_synapse = it_synapse_low; it_synapse != it_synapse_up; it_synapse++ )
					{
						double delay = *it_synapse;
						double synch = fabs(target_label - INTEGRATION_TIME - source_first_spike_time - delay);
						
						if (synch < best_synch)
						{
							best_synch = synch;
							it_best_synch_synapse = it_synapse;
						}
					}
				
					new_num_inputs[target_id] += 1;
						
					double weight = this->sample_Ge2e();
					double delay = *it_best_synch_synapse;
					
					
					// remove synapse
					unwired_delays.erase(it_best_synch_synapse);
					
					synapse_parameters.push_back(std::make_tuple(source_id, target_id, delay, weight));
					
					added_connections[source_id].insert(target_id);
					
					target_was_connected = true;	
					break;
				}
				
				
			} // end loop through source neurons
				
			// if target was connected by some source neuron, stop
			// reinsert target to a new place in the target array or
			// just delete it if number of inputs is exceeded
			candidates_by_input.erase(it_target);

			if ( target_was_connected )
			{
				if ( new_num_inputs[target_id] < max_num_inputs )
					candidates_by_input.insert(std::make_pair(new_num_inputs[target_id], std::make_pair(target_label, target_id)));	 
			}
			
		} // end loop through targets
		
		// if number of unwired synapses exceeds allowed amount, sample a new target
		int num_remaining_synapses = static_cast<int>(unwired_delays.size());
		
		if (num_remaining_synapses > allowed_num_lost)
		{
			// exit if network already contains N_RA neurons
			if (network_neurons.size() == N_RA) break;

			int num_to_sample = NUM_NEURONS_TO_SAMPLE;
			
			if ( num_remaining_synapses < NUM_NEURONS_TO_SAMPLE )
				num_to_sample = num_remaining_synapses;
				
			for (size_t i = 0; i < num_to_sample; i++)
			{
				// randomly sample source neuron
				int ind = noise_generator.sample_integer(0, new_source_first_spike_times_sorted.size()-1);
				int source_id = (*(new_source_first_spike_times_sorted.begin()+ind)).second;
			
				double source_first_spike_time = spikes_in_trial_soma_global[source_id][0];
				
				// randomly sample one of remaining synapses
				ind = noise_generator.sample_integer(0, unwired_delays.size()-1);
				
				std::vector<double>::iterator it_rand_synapse = unwired_delays.begin();
				std::advance(it_rand_synapse, ind);
				
				double delay = *it_rand_synapse;
				double label = source_first_spike_time + delay + INTEGRATION_TIME;
				
				// randomly sample a target which is not in the network
				std::vector<int> sampling_pool(N_RA);
				std::iota(sampling_pool.begin(), sampling_pool.end(), 0);
				
				int new_target_id;
				
				while (true)
				{
					int ind = noise_generator.sample_integer(0, sampling_pool.size()-1);
					new_target_id = sampling_pool[ind];
				
					if (network_neurons.find(new_target_id) == network_neurons.end())
						break;
					
					sampling_pool[ind] = sampling_pool.back();
					sampling_pool.pop_back();	
				}
				
				//std::cout << "New target id : " << new_target_id << std::endl;
				
				num_targets_sampled += 1;
				
				double weight = this->sample_Ge2e();
					
				syn_ID_RA_RA[source_id].push_back(new_target_id);
				weights_RA_RA[source_id].push_back(weight);
				axonal_delays_RA_RA[source_id].push_back(delay);
				 
				num_inputs[new_target_id] += 1;
							
				assigned_burst_labels[new_target_id] = label;
				
				candidate_targets.insert(std::pair<double,int>(label, new_target_id));												
				network_neurons.insert(new_target_id);
			
			
				// find synapse and delete from array with all source synapses
				std::vector<double>::iterator it = std::find(synapse_delays.begin(), synapse_delays.end(), delay);
				
				if (it != synapse_delays.end())
					synapse_delays.erase(it);
				else
					std::cerr << "Delay to erase was not found!" << std::endl;
					
				// delete synapse from sample array
				unwired_delays.erase(it_rand_synapse);	
			
				if (network_neurons.size() == N_RA) break;
			}
		}
				
		else
		{
			//std::cout << "Remaining unwired synapses: " << unwired_delays.size() << std::endl;
			break;
		}
		
		candidates_by_input.clear();
		// update candidate by inputs array
		for (std::multimap<double,int>::iterator it = candidate_targets.begin(); it != candidate_targets.end(); it++)
		{
			double target_label = (*it).first;
			int target_id = (*it).second;
				
			if ( num_inputs[target_id] < max_num_inputs )
				candidates_by_input.insert(std::make_pair(num_inputs[target_id], *it));	
			
		}
		
		//std::cout << "Remaining unwired synapses: " << unwired_delays.size() << std::endl;
		
		unwired_delays = synapse_delays;
		rewire_iteration++;
	} // end while there are synapses in the pool
	
	
	
	// update source arrays according to sampled synapses
	for (size_t i = 0; i < synapse_parameters.size(); i++)
	{
		int source_id = std::get<0>(synapse_parameters[i]);
		int target_id = std::get<1>(synapse_parameters[i]);
		
		double delay = std::get<2>(synapse_parameters[i]);
		double weight = std::get<3>(synapse_parameters[i]);
		
		
		
		//~ std::cout << source_id << " -> " << target_id 
				  //~ << " " << delay << " " << length << " " << weight
				  //~ << " " << std::get<0>(synapse_coord) << " " << std::get<1>(synapse_coord) << " " << std::get<2>(synapse_coord) << std::endl; 
		//~ 
		
		syn_ID_RA_RA[source_id].push_back(target_id);
		weights_RA_RA[source_id].push_back(weight);
		axonal_delays_RA_RA[source_id].push_back(delay);
		
		num_inputs[target_id] += 1;

	}
	
	
	for (auto it = new_source_first_spike_times_sorted.begin(); it < new_source_first_spike_times_sorted.end(); it++)
		indicators_connected_to_targets[(*it).second] = true;
	
	return 0;
}

void HvcNetwork::sample_network_without_RA2RA(int N_ra, int N_i, double pei, double pie, double mean_delay, double sd_delay)
{
	N_RA = N_ra;
	N_I = N_i;
	
	if (MPI_rank == 0){
		this->resize_global_arrays();
	
		// sample HVC(RA) -> HVC(I) connections
		for (int i = 0; i < N_RA; i++){	
			for (int j = 0; j < N_I; j++){
				if (noise_generator.random(1.0) < pei)
				{	
					double G = this->sample_Ge2i();

					weights_RA_I[i].push_back(G);
					syn_ID_RA_I[i].push_back(j);
				}
			}
			// sample axonal conduction delays
			int num_targets = static_cast<int>(syn_ID_RA_I[i].size());
			
			std::vector<double> delays(num_targets);
				
			if (mean_delay > 1e-3)
				delays = sample_axonal_delays_from_lognormal(num_targets, mean_delay, sd_delay);
		
			for (int j = 0; j < num_targets; j++)
				axonal_delays_RA_I[i].push_back(delays[j]);
		}
		
		// sample HVC(I) -> HVC(RA) connections
		for (int i = 0; i < N_I; i++){
			for (int j = 0; j < N_RA; j++){		
				if (noise_generator.random(1.0) < pie){
					double G = this->sample_Gi2e();

					weights_I_RA[i].push_back(G);
					syn_ID_I_RA[i].push_back(j);
				}
			}
			// sample axonal conduction delays
			int num_targets = static_cast<int>(syn_ID_I_RA[i].size());
			
			std::vector<double> delays(num_targets);
				
			if (mean_delay > 1e-3)
				delays = sample_axonal_delays_from_lognormal(num_targets, mean_delay, sd_delay);
		
			for (int j = 0; j < num_targets; j++)
				axonal_delays_I_RA[i].push_back(delays[j]);
		}		
	}
}

void HvcNetwork::remove_connections_to_neurons(int neuron_id)
{
	for (int i = 0; i < N_RA; i++)
	{
		if (syn_ID_RA_RA[i].size() > 0)
		{
			std::vector<int>::iterator it = std::find(syn_ID_RA_RA[i].begin(), syn_ID_RA_RA[i].end(), neuron_id);
			
			while (it != syn_ID_RA_RA[i].end())
			{
				int shift = std::distance(syn_ID_RA_RA[i].begin(), it);
				
				syn_ID_RA_RA[i][shift] = syn_ID_RA_RA[i].back();
				syn_ID_RA_RA[i].pop_back();
				
				axonal_delays_RA_RA[i][shift] = axonal_delays_RA_RA[i].back();
				axonal_delays_RA_RA[i].pop_back();
				
				weights_RA_RA[i][shift] = weights_RA_RA[i].back();
				weights_RA_RA[i].pop_back();
				
				it = std::find(syn_ID_RA_RA[i].begin(), syn_ID_RA_RA[i].end(), neuron_id);
			}
		}
	}
	
}

std::vector<double> HvcNetwork::sample_axonal_delays_from_lognormal(int N, double mean, double std)
{
	std::vector<double> delays(N);
	double sigma = sqrt(std::log(1 + std * std / (mean * mean) ));
	double mu = std::log(mean) - sigma * sigma / 2.0;
	
	for (int i = 0; i < N; i++)
		delays[i] = noise_generator.sample_lognormal_distribution(mu, sigma);
	
	return delays;
}

void HvcNetwork::sample_noise_based_on_dend_capacitance(const std::vector<double>& cm,
				std::vector<double>& mu_soma, std::vector<double>& std_soma, 
				std::vector<double>& mu_dend, std::vector<double>& std_dend){
	const std::vector<double> CM_DEND = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0};
	const std::vector<double> STD_DEND = {0.163, 0.198, 0.233, 0.263, 0.29, 0.315, 0.34, 0.36, 0.4, 0.42, 0.44, 0.46, 0.478, 0.495, 0.513, 0.53, 0.546, 0.562, 0.577, 0.592};

	const double std_soma_const = 0.1;
	const double mu_soma_const = 0.0;
	const double mu_dend_const = 0.0;

	assert(N_RA == static_cast<int>(cm.size()));

	mu_soma.resize(cm.size());
	std_soma.resize(cm.size());
	mu_dend.resize(cm.size());
	std_dend.resize(cm.size());
	
	std::fill(mu_soma.begin(), mu_soma.end(), mu_soma_const);
	std::fill(std_soma.begin(), std_soma.end(), std_soma_const);
	std::fill(mu_dend.begin(), mu_dend.end(), mu_dend_const);
	
	double dc = CM_DEND[1] - CM_DEND[0]; // capacitance resolution
	double c_min = CM_DEND.front();
	
	for (int i = 0; i < N_RA; i++){
		if ((cm[i] <= CM_DEND.front()) || (cm[i] >= CM_DEND.back()) ){
			std::cout << "Capacitance not in range: cm[" << i << "] = " << cm[i] << std::endl;
			return;
		}
	
		int ind_floor = static_cast<int>(floor((cm[i]-c_min) / dc));
		int ind_ceil = static_cast<int>(ceil((cm[i]-c_min) / dc));
		
		double alpha = (cm[i] - CM_DEND[ind_floor]) / dc;
		
		double std_dend_neuron = (1-alpha) * STD_DEND[ind_floor] 
							+ alpha * STD_DEND[ind_ceil];
							
		std_dend[i] = std_dend_neuron;
	}
}

//~ void HvcNetwork::sample_capacitance_and_integration_times_lognormal(int N, std::pair<double,double> c_range, double c_mean, double c_std,
						//~ std::vector<double>& capacitance_dend, std::vector<double>& integration_times){
	//~ const std::vector<double> CM_DEND = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0};
	//~ const std::vector<double> INTEGRATION_TIMES = {2.845373, 3.2756732, 3.6486728, 3.978773, 4.279573, 4.557473, 4.8193727, 5.0693727, 5.3418775, 5.5762773, 5.8042774, 6.0254774, 6.2410774, 6.4510775, 6.655478, 6.857078, 7.054678, 7.2482777, 7.439078, 7.6302776, 7.8170776, 8.002277, 8.185078, 8.367078, 8.549878, 8.727878, 8.904677, 9.083878, 9.259877, 9.436678, 9.611878, 9.785877, 9.962277, 10.1322775, 10.304678, 10.477878, 10.646677, 10.814679, 10.980678, 11.145077, 11.306679, 11.466278, 11.624277, 11.782278, 11.938678, 12.090277, 12.243477, 12.395078, 12.5458765, 12.693878, 12.843078, 12.990678, 13.139876, 13.289878, 13.43508, 13.585078, 13.732279, 13.880279, 14.027877, 14.177478, 14.325479, 14.475477, 14.625479, 14.777477, 14.928677, 15.080679, 15.232677, 15.387478, 15.542678, 15.697879, 15.855477, 16.014277, 16.174679, 16.333477, 16.497078, 16.660677, 16.826677, 16.995476, 17.164278, 17.333477, 17.507477, 17.68348, 17.859877, 18.037077, 18.220278, 18.404678, 18.589876, 18.781076, 18.972279, 19.168678, 19.365078, 19.565477, 19.77188, 19.981878, 20.192678, 20.408278, 20.630278, 20.855078, 21.084278};

	//~ double c_min_dist = c_range.first;
	//~ double c_max_dist = c_range.second;
	
	//~ if (c_max_dist > CM_DEND.back()){
		//~ std::cout << "Max capacitance in sample_integration_times exceeds max in the simulated data!" << std::endl;
		//~ return;
	//~ }
	
	//~ if (c_min_dist < CM_DEND.front()){
		//~ std::cout << "Min capacitance time in sample_integration_times is below min in the simulated data!" << std::endl;
		//~ return;
	//~ }

	//~ integration_times.resize(N);
	//~ capacitance_dend.resize(N);
	
	//~ double sigma = sqrt(std::log(1 + c_std * c_std / (c_mean * c_mean) ));
	//~ double mu = std::log(c_mean) - sigma * sigma / 2.0;
	
	
	//~ for (int i = 0; i < N; i++){
		//~ double rand_c = noise_generator.sample_lognormal_distribution(mu, sigma);
		
		
		//~ while ((rand_c <= c_min_dist)||(rand_c >= c_max_dist)) rand_c = noise_generator.sample_lognormal_distribution(mu, sigma);
		
		//~ auto it_low = std::lower_bound(CM_DEND.begin(), CM_DEND.end(), rand_c);
		//~ auto it_up = std::upper_bound(CM_DEND.begin(), CM_DEND.end(), rand_c);
		
		//~ assert(it_low != CM_DEND.begin());
		//~ assert(it_up != CM_DEND.end());
		
		//~ double alpha = (rand_c - *(it_low-1)) / (*it_up - *(it_low-1));
		
		//~ //std::cout << *(it_low-1) << ", " << rand_int << ", " << *(it_up) << ", " << alpha << std::endl;
		
		//~ capacitance_dend[i] = rand_c;
		
		//~ integration_times[i] = (1-alpha) * INTEGRATION_TIMES[std::distance(CM_DEND.begin(), it_low-1)]
							//~ + alpha * INTEGRATION_TIMES[std::distance(CM_DEND.begin(), it_up)];
	//~ }
//~ }


void HvcNetwork::sample_capacitance_and_integration_times_lognormal(int N, std::pair<double,double> int_range, double int_mean, double int_std,
						std::vector<double>& capacitance_dend, std::vector<double>& integration_times){
	// gee_max = 0.004
	//const std::vector<double> CM_DEND = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0};
	//const std::vector<double> INTEGRATION_TIMES = {2.845373, 3.2756732, 3.6486728, 3.978773, 4.279573, 4.557473, 4.8193727, 5.0693727, 5.3418775, 5.5762773, 5.8042774, 6.0254774, 6.2410774, 6.4510775, 6.655478, 6.857078, 7.054678, 7.2482777, 7.439078, 7.6302776, 7.8170776, 8.002277, 8.185078, 8.367078, 8.549878, 8.727878, 8.904677, 9.083878, 9.259877, 9.436678, 9.611878, 9.785877, 9.962277, 10.1322775, 10.304678, 10.477878, 10.646677, 10.814679, 10.980678, 11.145077, 11.306679, 11.466278, 11.624277, 11.782278, 11.938678, 12.090277, 12.243477, 12.395078, 12.5458765, 12.693878, 12.843078, 12.990678, 13.139876, 13.289878, 13.43508, 13.585078, 13.732279, 13.880279, 14.027877, 14.177478, 14.325479, 14.475477, 14.625479, 14.777477, 14.928677, 15.080679, 15.232677, 15.387478, 15.542678, 15.697879, 15.855477, 16.014277, 16.174679, 16.333477, 16.497078, 16.660677, 16.826677, 16.995476, 17.164278, 17.333477, 17.507477, 17.68348, 17.859877, 18.037077, 18.220278, 18.404678, 18.589876, 18.781076, 18.972279, 19.168678, 19.365078, 19.565477, 19.77188, 19.981878, 20.192678, 20.408278, 20.630278, 20.855078, 21.084278};

	// gee_max = 0.032
	const std::vector<double> CM_DEND = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0};
	const std::vector<double> INTEGRATION_TIMES = {2.5364225, 2.8600223, 3.1576223, 3.4440224, 3.7118223, 3.9514225, 4.1682224, 4.3666224, 4.5574226, 4.7392225, 4.920422, 5.0968227, 5.2706227, 5.441422, 5.607622, 5.7696214, 5.9262223, 6.077622, 6.2248225, 6.367223};

	double int_min_dist = int_range.first;
	double int_max_dist = int_range.second;
	
	if (int_max_dist > INTEGRATION_TIMES.back()){
		std::cout << "Max integration time in sample_integration_times exceeds max in the simulated data!" << std::endl;
		return;
	}
	
	if (int_min_dist < INTEGRATION_TIMES.front()){
		std::cout << "Min integration time in sample_integration_times is below min in the simulated data!" << std::endl;
		return;
	}

	integration_times.resize(N);
	capacitance_dend.resize(N);
	
	double sigma = sqrt(std::log(1 + int_std * int_std / (int_mean * int_mean) ));
	double mu = std::log(int_mean) - sigma * sigma / 2.0;
	
	
	for (int i = 0; i < N; i++){
		double rand_int = noise_generator.sample_lognormal_distribution(mu, sigma);
		
		
		while ((rand_int <= int_min_dist)||(rand_int >= int_max_dist)) rand_int = noise_generator.sample_lognormal_distribution(mu, sigma);
		
		auto it_low = std::lower_bound(INTEGRATION_TIMES.begin(), INTEGRATION_TIMES.end(), rand_int);
		auto it_up = std::upper_bound(INTEGRATION_TIMES.begin(), INTEGRATION_TIMES.end(), rand_int);
		
		assert(it_low != INTEGRATION_TIMES.begin());
		assert(it_up != INTEGRATION_TIMES.end());
		
		double alpha = (rand_int - *(it_low-1)) / (*it_up - *(it_low-1));
		
		//std::cout << *(it_low-1) << ", " << rand_int << ", " << *(it_up) << ", " << alpha << std::endl;
		
		capacitance_dend[i] = (1-alpha) * CM_DEND[std::distance(INTEGRATION_TIMES.begin(), it_low-1)]
							+ alpha * CM_DEND[std::distance(INTEGRATION_TIMES.begin(), it_up)];
							
		integration_times[i] = rand_int;
	}
}


void HvcNetwork::sample_capacitance_and_integration_times_uniform(int N, std::pair<double,double> int_range, 
						std::vector<double>& capacitance_dend, std::vector<double>& integration_times){
	const std::vector<double> CM_DEND = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0};
	const std::vector<double> INTEGRATION_TIMES = {2.845373, 3.2756732, 3.6486728, 3.978773, 4.279573, 4.557473, 4.8193727, 5.0693727, 5.3418775, 5.5762773, 5.8042774, 6.0254774, 6.2410774, 6.4510775, 6.655478, 6.857078, 7.054678, 7.2482777, 7.439078, 7.6302776, 7.8170776, 8.002277, 8.185078, 8.367078, 8.549878, 8.727878, 8.904677, 9.083878, 9.259877, 9.436678, 9.611878, 9.785877, 9.962277, 10.1322775, 10.304678, 10.477878, 10.646677, 10.814679, 10.980678, 11.145077, 11.306679, 11.466278, 11.624277, 11.782278, 11.938678, 12.090277, 12.243477, 12.395078, 12.5458765, 12.693878, 12.843078, 12.990678, 13.139876, 13.289878, 13.43508, 13.585078, 13.732279, 13.880279, 14.027877, 14.177478, 14.325479, 14.475477, 14.625479, 14.777477, 14.928677, 15.080679, 15.232677, 15.387478, 15.542678, 15.697879, 15.855477, 16.014277, 16.174679, 16.333477, 16.497078, 16.660677, 16.826677, 16.995476, 17.164278, 17.333477, 17.507477, 17.68348, 17.859877, 18.037077, 18.220278, 18.404678, 18.589876, 18.781076, 18.972279, 19.168678, 19.365078, 19.565477, 19.77188, 19.981878, 20.192678, 20.408278, 20.630278, 20.855078, 21.084278};

	double int_min_dist = int_range.first;
	double int_max_dist = int_range.second;
	
	if (int_max_dist > INTEGRATION_TIMES.back()){
		std::cout << "Max integration time in sample_integration_times exceeds max in the simulated data!" << std::endl;
		return;
	}
	
	if (int_min_dist < INTEGRATION_TIMES.front()){
		std::cout << "Min integration time in sample_integration_times is below min in the simulated data!" << std::endl;
		return;
	}

	integration_times.resize(N);
	capacitance_dend.resize(N);
	
	for (int i = 0; i < N; i++){
		double rand_int = int_min_dist + noise_generator.random(int_max_dist-int_min_dist);
		
		auto it_low = std::lower_bound(INTEGRATION_TIMES.begin(), INTEGRATION_TIMES.end(), rand_int);
		auto it_up = std::upper_bound(INTEGRATION_TIMES.begin(), INTEGRATION_TIMES.end(), rand_int);
		
		assert(it_low != INTEGRATION_TIMES.begin());
		assert(it_up != INTEGRATION_TIMES.end());
		
		double alpha = (rand_int - *(it_low-1)) / (*it_up - *(it_low-1));
		
		capacitance_dend[i] = (1-alpha) * CM_DEND[std::distance(INTEGRATION_TIMES.begin(), it_low-1)]
							+ alpha * CM_DEND[std::distance(INTEGRATION_TIMES.begin(), it_up)];
							
		integration_times[i] = rand_int;
	}
}

double HvcNetwork::sample_Ge2i()
{
    return noise_generator.random(Gei_max);
}

double HvcNetwork::sample_Gi2e()
{
    return noise_generator.random(Gie_max);
}

double HvcNetwork::sample_Ge2e()
{
    return noise_generator.random(Gee_max);
}

void HvcNetwork::distribute_work()
{
	// distribute the work between processes for sampling connections and axonal delays
	N_RA_sizes.resize(MPI_size); // array with number of RA neurons per process
	N_I_sizes.resize(MPI_size); // array with number of I neurons per process

	for (int i = 0; i < MPI_size; i++)
	{
		N_RA_sizes[i] = N_RA / MPI_size;
		N_I_sizes[i] = N_I / MPI_size;
	}
	int RA_remain = N_RA % MPI_size;
	int I_remain = N_I % MPI_size;
	int j = 0;

	// distribute RA neurons
	while (RA_remain > 0)
	{
		N_RA_sizes[j] += 1;
		RA_remain -= 1;
		j += 1;

		if (j >= MPI_size)
			j -= MPI_size;
	}

	// distribute I neurons
	j = 0;
	
	while (I_remain > 0)
	{	
		N_I_sizes[j] += 1;
		I_remain -= 1;
		j += 1;

		if (j >= MPI_size)
			j -= MPI_size;
	}

	N_RA_local = N_RA_sizes[MPI_rank]; // assign number of RA neurons for each process
	N_I_local = N_I_sizes[MPI_rank]; // assign number of I neurons for each process

	//printf("My rank is %d; N_RA_local = %d; N_I_local = %d\n", MPI_rank, N_RA_local, N_I_local);
	
	Id_RA_local.resize(N_RA_local);
	Id_I_local.resize(N_I_local);
	
	 // assign real id to neurons
	for (int i = 0; i < N_RA_local; i++)
	{
		// assign real Id for RA neurons
		int N = 0; // number of neurons in the processes with lower rank
		
		for (int k = 0; k < MPI_rank; k++)
			N += N_RA_sizes[k];

		Id_RA_local[i] = N + i;
	}
	
    // assign real ID for I neurons
    for (int i = 0; i < N_I_local; i++)
	{
		int N = 0; // number of neurons in the processes with lower rank
		
		for (int k = 0; k < MPI_rank; k++)
			N += N_I_sizes[k];

		Id_I_local[i] = N + i;
	}
}

void HvcNetwork::prepare_slaves_for_testing(std::string fileTraining)
{
	if (MPI_rank == 0)
		this->read_training_neurons(fileTraining.c_str());
	
	// send number of neurons to all processes
	MPI_Bcast(&N_RA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N_TR, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N_I, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	// send training neurons
	if (MPI_rank != 0)
		training_neurons.resize(N_TR);
	
	MPI_Bcast(&training_neurons[0], N_TR, MPI_INT, 0, MPI_COMM_WORLD);
	
	this->initialize_dynamics();
	this->scatter_connections_betweenRAandI();
	this->scatter_connections_RA2RA();
	
	//if (MPI_rank == 0)
	//	std::cout << "Slaves are prepared for testing!\n" << std::endl; 
}

void HvcNetwork::scale_weights(double scale){
	for (int i = 0; i < N_RA_local; i++)
		for (size_t j = 0; j < weights_RA_RA_local[i].size(); j++)
			weights_RA_RA_local[i][j] *= scale;
}

void HvcNetwork::initialize_dynamics()
{	
	this->distribute_work();
	
	this->resize_arrays_for_I(N_I_local, N_I);
	this->resize_arrays_for_RA(N_RA_local, N_RA);
	
	this->set_noise();
	this->set_dynamics();				       

	//if (MPI_rank == 0)
	//	std::cout << "Dynamics is initialized!\n" << std::endl; 
}

void HvcNetwork::set_noise_based_on_dend_capacitance(const std::vector<double>& cm){
	const std::vector<double> CM_DEND = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0};
	const std::vector<double> std_dend = {0.163, 0.198, 0.233, 0.263, 0.29, 0.315, 0.34, 0.36, 0.4, 0.42, 0.44, 0.46, 0.478, 0.495, 0.513, 0.53, 0.546, 0.562, 0.577, 0.592};

	double std_soma = 0.1;
	double mu_soma = 0.0;
	double mu_dend = 0.0;

	double dc = CM_DEND[1] - CM_DEND[0]; // capacitance resolution
	
	double c_min = CM_DEND.front();
	
	assert(N_RA_local == static_cast<int>(cm.size()));
	
	for (int i = 0; i < N_RA_local; i++){
		if ((cm[i] <= CM_DEND.front()) || (cm[i] >= CM_DEND.back()) ){
			std::cout << "Capacitance not in range: cm[" << i << "] = " << cm[i] << std::endl;
			return;
		}
		
		int ind_floor = static_cast<int>(floor((cm[i]-c_min) / dc));
		int ind_ceil = static_cast<int>(ceil((cm[i]-c_min) / dc));
		
		double alpha = (cm[i] - CM_DEND[ind_floor]) / dc;
		
		double std_dend_neuron = (1-alpha) * std_dend[ind_floor] 
							+ alpha * std_dend[ind_ceil];
	
		if (std_dend_neuron > 0.3)
			std::cout << "cm, std, ind_floor, ind_ceil, alpha = " << cm[i] << ", " << std_dend_neuron 
				      << ", " << ind_floor << ", " << ind_ceil << ", " << alpha << std::endl;
							
		HVCRA_local[i].set_white_noise(mu_soma, std_soma, mu_dend, std_dend_neuron);
	}
}

void HvcNetwork::set_noise()
{
	//std::cout << "Rank " << MPI_rank << " in set_noise; N_RA_local = " << N_RA_local << " ; HVCRA_local.size = " << HVCRA_local.size() 
	//                                             << " ; N_I_local = " << N_I_local << " ; HVCI_local.size = " << HVCI_local.size() << std::endl;
	
	for (int i = 0; i < N_RA_local; i++)
	{	
		HVCRA_local[i].set_noise_generator(&noise_generator);
		HVCRA_local[i].set_white_noise(WHITE_NOISE_MEAN_SOMA, WHITE_NOISE_STD_SOMA, WHITE_NOISE_MEAN_DEND, WHITE_NOISE_STD_DEND);
	}

    for (int i = 0; i < N_I_local; i++)
	{
        HVCI_local[i].set_noise_generator(&noise_generator);
        HVCI_local[i].set_poisson_noise();
    }
}

void HvcNetwork::set_dynamics()
{
	for (int i = 0; i < N_RA_local; i++)
		HVCRA_local[i].set_dynamics(TIMESTEP);

	
    for (int i = 0; i < N_I_local; i++)
		HVCI_local[i].set_dynamics(TIMESTEP);
}

void HvcNetwork::resample_strength_of_connections_in_network()
{
	// resample HVC(I) -> HVC(RA) connections strength
	for (int i = 0; i < syn_ID_I_RA.size(); i++)
		for (size_t j = 0; j < syn_ID_I_RA[i].size(); j++)
			weights_I_RA[i][j] = this->sample_Gi2e();
			
	// resample HVC(RA) -> HVC(I) connections strength
	for (int i = 0; i < syn_ID_RA_I.size(); i++)
	{
		for (size_t j = 0; j < syn_ID_RA_I[i].size(); j++)
			weights_RA_I[i][j] = this->sample_Ge2i();
	}
	
	// resample HVC(RA) -> HVC(RA) connections strength
	for (int i = 0; i < syn_ID_RA_RA.size(); i++)
	{	
		for (size_t j = 0; j < syn_ID_RA_RA[i].size(); j++)
			weights_RA_RA[i][j] = this->sample_Ge2e();
	}
}

void HvcNetwork::set_connection_strengths(double gee_max, double gei_max, double gie_max)
{
	Gee_max = gee_max;
	Gei_max = gei_max;
	Gie_max = gie_max;
}

void HvcNetwork::initialize_generators(unsigned seed)
{   
    noise_generator.set_seed(seed + 1000 * MPI_rank);
}

void HvcNetwork::scatter_connections_betweenRAandI()
{
	int *sendcounts_RA = new int[MPI_size];
    int *displs_RA = new int[MPI_size];
	
	int *sendcounts_I = new int[MPI_size];
    int *displs_I = new int[MPI_size];

	sendcounts_RA[0] = N_RA_sizes[0];
	displs_RA[0] = 0;

	for (int i = 1; i < MPI_size; i++)
	{
		sendcounts_RA[i] = N_RA_sizes[i];
		displs_RA[i] = displs_RA[i-1] + sendcounts_RA[i-1];
	}
	
	sendcounts_I[0] = N_I_sizes[0];
	displs_I[0] = 0;

	for (int i = 1; i < MPI_size; i++)
	{
		sendcounts_I[i] = N_I_sizes[i];
		displs_I[i] = displs_I[i-1] + sendcounts_I[i-1];
	}

	std::vector<int> num_targets_RA_I_global; // global number of interneuron targets for HVC(RA) source neuron
	std::vector<int> num_targets_RA_I_local(N_RA_local); // local number of interneuron targets for HVC(RA) source neuron
	
	std::vector<int> num_targets_I_RA_global; // global number of HVC(I) targets for HVC(RA) source neuron
	std::vector<int> num_targets_I_RA_local(N_I_local); // local number of HVC(I) targets for HVC(RA) source neuron
	
	// prepare global arrays with number of targets for each neuron
	if (MPI_rank == 0)
	{	
		num_targets_RA_I_global.resize(N_RA);
		num_targets_I_RA_global.resize(N_I);
		
		for (int i = 0; i < N_RA; i++)
			num_targets_RA_I_global[i] = static_cast<int>(syn_ID_RA_I[i].size());		
			
		for (int i = 0; i < N_I; i++)
			num_targets_I_RA_global[i] = static_cast<int>(syn_ID_I_RA[i].size());
	}
	
	// send number of targets to all processes
	MPI_Scatterv(&num_targets_RA_I_global[0], sendcounts_RA, displs_RA, MPI_INT, 
					&num_targets_RA_I_local[0], N_RA_local, MPI_INT, 0, MPI_COMM_WORLD);
	
	MPI_Scatterv(&num_targets_I_RA_global[0], sendcounts_I, displs_I, MPI_INT, 
					&num_targets_I_RA_local[0], N_I_local, MPI_INT, 0, MPI_COMM_WORLD);
	
	delete[] sendcounts_RA;
	delete[] displs_RA;
	delete[] sendcounts_I;
	delete[] displs_I;
	
	// resize all local arrays
	weights_RA_I_local.resize(N_RA_local);
	syn_ID_RA_I_local.resize(N_RA_local);
	axonal_delays_RA_I_local.resize(N_RA_local);
	
	weights_I_RA_local.resize(N_RA_local);
	syn_ID_I_RA_local.resize(N_RA_local);
	axonal_delays_I_RA_local.resize(N_I_local);
	
	for (int i = 0; i < N_RA_local; i++)
	{
		weights_RA_I_local[i].resize(num_targets_RA_I_local[i]);
		syn_ID_RA_I_local[i].resize(num_targets_RA_I_local[i]);	
		axonal_delays_RA_I_local[i].resize(num_targets_RA_I_local[i]);	
	}
	
	for (int i = 0; i < N_I_local; i++)
	{
		weights_I_RA_local[i].resize(num_targets_I_RA_local[i]);
		syn_ID_I_RA_local[i].resize(num_targets_I_RA_local[i]);
		axonal_delays_I_RA_local[i].resize(num_targets_I_RA_local[i]);
	}
	// now send all connections from master process to other processes
	
	MPI_Status status;
	
	if (MPI_rank == 0)
	{
		// copy global data from the master process to local arrays
		for (int i = 0; i < N_RA_local; i++)
		{
			weights_RA_I_local[i] = weights_RA_I[i];
			syn_ID_RA_I_local[i] = syn_ID_RA_I[i];
			axonal_delays_RA_I_local[i] = axonal_delays_RA_I[i];
		}
		
		for (int i = 0; i < N_I_local; i++)
		{
			weights_I_RA_local[i] = weights_I_RA[i];
			syn_ID_I_RA_local[i] = syn_ID_I_RA[i];
			axonal_delays_I_RA_local[i] = axonal_delays_I_RA[i];
		}
		
		int indRA = N_RA_sizes[0]; // number of RA neurons in the processes with lower rank
		int indI = N_I_sizes[0]; // number of RA neurons in the processes with lower rank
		
		MPI_Status status; // status of MPI Recv communication call
		
		for (int i = 1; i < MPI_size; i++)
		{
			for (int j = 0; j < N_RA_sizes[i]; j++)
			{
				int send_index = indRA + j;
				
				// send HVC(RA) -> HVC(I) connections
				MPI_Send(&weights_RA_I[send_index][0], 
									static_cast<int>(weights_RA_I[send_index].size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				
				MPI_Send(&syn_ID_RA_I[send_index][0], 
									static_cast<int>(syn_ID_RA_I[send_index].size()), MPI_INT, i, 0, MPI_COMM_WORLD);
				
				MPI_Send(&axonal_delays_RA_I[send_index][0], 
								static_cast<int>(axonal_delays_RA_I[send_index].size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
			
			for (int j = 0; j < N_I_sizes[i]; j++)
			{
				int send_index = indI + j;

				// send HVC(I) -> HVC(RA) connections
				MPI_Send(&weights_I_RA[send_index][0], 
									static_cast<int>(weights_I_RA[send_index].size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				
				MPI_Send(&syn_ID_I_RA[send_index][0], 
									static_cast<int>(syn_ID_I_RA[send_index].size()), MPI_INT, i, 0, MPI_COMM_WORLD);
				
				MPI_Send(&axonal_delays_I_RA[send_index][0], 
									static_cast<int>(axonal_delays_I_RA[send_index].size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			
			}
			
			indRA += N_RA_sizes[i];
			indI += N_I_sizes[i];
			
		}
	}

    else
    {
        for (int i = 0; i < N_RA_local; i++)
        {
			// receive HVC(RA) -> HVC(I) connections
			MPI_Recv(&weights_RA_I_local[i][0],
									num_targets_RA_I_local[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
			
			MPI_Recv(&syn_ID_RA_I_local[i][0],
									num_targets_RA_I_local[i], MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
									
			MPI_Recv(&axonal_delays_RA_I_local[i][0],
									num_targets_RA_I_local[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        }
        
        for (int i = 0; i < N_I_local; i++)
        {
			// receive HVC(I) -> HVC(RA) connections
            MPI_Recv(&weights_I_RA_local[i][0],
									num_targets_I_RA_local[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
			
			MPI_Recv(&syn_ID_I_RA_local[i][0],
									num_targets_I_RA_local[i], MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
									
			MPI_Recv(&axonal_delays_I_RA_local[i][0],
									num_targets_I_RA_local[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		}	
    }	
}

void HvcNetwork::scatter_connections_RA2RA()
{
	int *sendcounts_RA = new int[MPI_size];
    int *displs_RA = new int[MPI_size];
	
	sendcounts_RA[0] = N_RA_sizes[0];
	displs_RA[0] = 0;

	for (int i = 1; i < MPI_size; i++)
	{
		sendcounts_RA[i] = N_RA_sizes[i];
		displs_RA[i] = displs_RA[i-1] + sendcounts_RA[i-1];
	}
	
	std::vector<int> num_targets_RA_RA_global; // global number of HVC(RA) targets for HVC(RA) source neuron
	std::vector<int> num_targets_RA_RA_local(N_RA_local); // local number of HVC(RA) targets for HVC(RA) source neuron
	
	// prepare global arrays with number of targets for each neuron
	if (MPI_rank == 0)
	{	
		num_targets_RA_RA_global.resize(N_RA);
		
		for (int i = 0; i < N_RA; i++)
			num_targets_RA_RA_global[i] = static_cast<int>(syn_ID_RA_RA[i].size());		
	}
	
	// send number of targets to all processes
	MPI_Scatterv(&num_targets_RA_RA_global[0], sendcounts_RA, displs_RA, MPI_INT, 
					&num_targets_RA_RA_local[0], N_RA_local, MPI_INT, 0, MPI_COMM_WORLD);
	
	delete[] sendcounts_RA;
	delete[] displs_RA;
	
	// resize all local arrays
	weights_RA_RA_local.resize(N_RA_local);
	syn_ID_RA_RA_local.resize(N_RA_local);
	axonal_delays_RA_RA_local.resize(N_RA_local);
	
	for (int i = 0; i < N_RA_local; i++)
	{	
		weights_RA_RA_local[i].resize(num_targets_RA_RA_local[i]);
		syn_ID_RA_RA_local[i].resize(num_targets_RA_RA_local[i]);	
		axonal_delays_RA_RA_local[i].resize(num_targets_RA_RA_local[i]);	
	}
	
	// now send all connections from master process to other processes
	MPI_Status status;
	
	if (MPI_rank == 0)
	{
		// copy global data from the master process to local arrays
		for (int i = 0; i < N_RA_local; i++)
		{
			weights_RA_RA_local[i] = weights_RA_RA[i];
			syn_ID_RA_RA_local[i] = syn_ID_RA_RA[i];
			axonal_delays_RA_RA_local[i] = axonal_delays_RA_RA[i];
		}
		
		int indRA = N_RA_sizes[0]; // number of RA neurons in the processes with lower rank
		
		MPI_Status status; // status of MPI Recv communication call
		
		for (int i = 1; i < MPI_size; i++)
		{
			for (int j = 0; j < N_RA_sizes[i]; j++)
			{
				int send_index = indRA + j;
				
				// send HVC(RA) -> HVC(RA) connections
				MPI_Send(&weights_RA_RA[send_index][0], 
									static_cast<int>(weights_RA_RA[send_index].size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				
				MPI_Send(&syn_ID_RA_RA[send_index][0], 
									static_cast<int>(syn_ID_RA_RA[send_index].size()), MPI_INT, i, 0, MPI_COMM_WORLD);
				
				MPI_Send(&axonal_delays_RA_RA[send_index][0], 
								static_cast<int>(axonal_delays_RA_RA[send_index].size()), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
			
			indRA += N_RA_sizes[i];	
		}
	}

    else
    {
        for (int i = 0; i < N_RA_local; i++)
        {						
			// receive HVC(RA) -> HVC(RA) connections
			MPI_Recv(&weights_RA_RA_local[i][0],
									num_targets_RA_RA_local[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
			
			MPI_Recv(&syn_ID_RA_RA_local[i][0],
									num_targets_RA_RA_local[i], MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
									
			MPI_Recv(&axonal_delays_RA_RA_local[i][0],
									num_targets_RA_RA_local[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        }	
    }	
}


void HvcNetwork::randomize_after_trial()
{
    for (int i = 0; i < N_RA; i++)
    {
        std::vector<double>().swap(spikes_in_trial_soma_global[i]);
        std::vector<double>().swap(spikes_in_trial_dend_global[i]);

    }

    for (int i = 0; i < N_I; i++)
        std::vector<double>().swap(spikes_in_trial_interneuron_global[i]);

    for (int i = 0; i < N_RA_local; i++)
    {
	    HVCRA_local[i].set_to_rest();

        std::vector<double>().swap(spikes_in_trial_soma_local[i]);
        std::vector<double>().swap(spikes_in_trial_dend_local[i]);    
    }

    for (int i = 0; i < N_I_local; i++)
    {
        HVCI_local[i].set_to_rest();
        std::vector<double>().swap(spikes_in_trial_interneuron_local[i]);
    }
}

void HvcNetwork::set_training_conductance_pulse()
{
    for (int i = 0; i < N_TR; i++)
	{
		int rank;
		int shift;
		
		this->get_neuronRA_location(training_neurons[i], &rank, &shift);
		
		if (MPI_rank == rank)
			HVCRA_local[shift].raiseE(CONDUCTANCE_PULSE);
	}
}

void HvcNetwork::run_polychronous_network(double trial_duration)
{	
	double training_spread = 0.0;
	bool record_excitatory_current = false;
	
	this->trial_distributed_training(trial_duration, training_spread, record_excitatory_current);
	this->gather_data();
}

void HvcNetwork::test_network(int num_trials, double trial_duration, double spread, bool record, std::string outputDirectory, std::string fileSimName)
{	
	bool record_excitatory_current = true; // track excitatory currents of HVC-RA neurons
	std::vector<double> max_excitatory_currents; // array with max excitatory currents of HVC-RA neurons	
   
	std::vector<std::vector<double>> average_burst_onset_time; // array with average dendritic spike time in every trial
	std::vector<std::vector<int>> num_dendritic_spikes_in_trials; // number of dendritic spikes produced in all trials
	std::vector<std::vector<int>> num_somatic_spikes_in_trials; // number of somatic spikes produced in all trials

	average_burst_onset_time.resize(N_RA);
    num_dendritic_spikes_in_trials.resize(N_RA);
    num_somatic_spikes_in_trials.resize(N_RA);

    if (MPI_rank == 0)
    {
        for (int j = 0; j < N_RA; j++)
        {
            num_dendritic_spikes_in_trials[j].resize(num_trials);
            num_somatic_spikes_in_trials[j].resize(num_trials);
        }
    }
   
	////////////////////////////////
	// Record from random neurons //
	////////////////////////////////
	
	int num_RA_to_write = 0;
	
	if (record)
		num_RA_to_write = 200;
	
    std::vector<int> RAtoWrite(num_RA_to_write);

	if (MPI_rank == 0)
	{
		std::vector<int> not_training(N_RA - N_TR);
		
		int not_training_id = 0;
		for (int i = 0; i < N_RA; i++)
			if (std::find(training_neurons.begin(), training_neurons.end(), i) == training_neurons.end())
			{
				not_training[not_training_id] = i;
				not_training_id += 1;
			}

		
		RAtoWrite = sample_randomly_noreplacement_from_array(not_training, num_RA_to_write, noise_generator);
									
		std::cout << "Training neurons:\n"; 
		for (int training : training_neurons) std::cout << training << " ";
		std::cout << "\n" << std::endl;
		
		//std::cout << "HVC(RA) neurons to write:\n"; 
		//for (int towrite : RAtoWrite) std::cout << towrite << " ";
		//std::cout << "\n" << std::endl;
	}
	
	MPI_Bcast(&RAtoWrite[0], num_RA_to_write, MPI_INT, 0, MPI_COMM_WORLD);
									
    std::vector<int> ItoWrite = {};
    
    //~ ////////////////////////////
    //~ // Record from all HVC-RA //
    //~ ////////////////////////////
    //~ std::vector<int> RAtoWrite = {};
//~ 
    //~ for (int i = 0; i < N_RA; i++)
        //~ RAtoWrite.push_back(i);
        //~ 
    //~ // generate excitation times for training neurons
    //~ std::vector<double> injection_times(N_RA);
    //~ 
    //~ if (MPI_rank == 0)
    //~ {
		//~ 
		//~ for (int i = 0; i < N_RA; i++)
			//~ injection_times[i] = WAITING_TIME - INJECTION_TIME_SPREAD / 2.0 + noise_generator.random(INJECTION_TIME_SPREAD);
		//~ 
		//~ 
		//~ std::cout << "Injection times for training neurons: \n";
		//~ for (int i = 0; i < N_TR; i++)
			//~ std::cout << "Neuron " << training_neurons[i] << " injection_time = " << injection_times[training_neurons[i]] << "\n";
		//~ std::cout << std::endl;
	//~ }
	//~ 
	//~ MPI_Bcast(&injection_times[0], N_RA, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < num_trials; i++)
	{
		
        if (MPI_rank == 0)
            std::cout << "Trial " << i << std::endl;

		// record membrane potentials of neurons from the last trial
		//if (i == num_trials - 1)
		//{
			for (size_t j = 0; j < RAtoWrite.size(); j++)
			{
				std::string fileNeuron = outputDirectory + fileSimName + "trial" + std::to_string(i) + "_RA" + std::to_string(RAtoWrite[j]) + ".bin";
				
				int shift;
				int rank;
				
				this->get_neuronRA_location(RAtoWrite[j], &rank, &shift);
				
				if (MPI_rank == rank)
					HVCRA_local[shift].set_recording(fileNeuron);
			}
			
			for (size_t j = 0; j < ItoWrite.size(); j++)
			{
				std::string fileNeuron = outputDirectory + fileSimName + "trial" + std::to_string(i) + "_I" + std::to_string(ItoWrite[j]) + ".bin";
				
				int shift;
				int rank;
				
				this->get_neuronI_location(ItoWrite[j], &rank, &shift);
				
				if (MPI_rank == rank)
					HVCI_local[shift].set_recording(fileNeuron);
			}
			
		//}
		
		max_excitatory_currents = this->trial_distributed_training(trial_duration, spread, record_excitatory_current);
		
		this->gather_data();
		
		
		if (MPI_rank == 0)
		{
			// spikes and bursts of HVC-RA neurons
			
			size_t num_HVCRA_spikes = 0; // total number of spikes of HVC-RA neurons during trial
			size_t num_HVCRA_bursts = 0; // total number of bursts of HVC-RA neurons during trial
			size_t num_HVCRA_silent = 0; // number of HVC-RA neurons that did not burst
			
			for (int i = 0; i < N_RA; i++)
			{
				if ( spikes_in_trial_soma_global[i].size() >= 3 )
					//printf("Average dendritic spike time = %f\n", average_spike_time);
					average_burst_onset_time[i].push_back(spikes_in_trial_soma_global[i][0]);
				else
					num_HVCRA_silent += 1;
				
				num_HVCRA_spikes += spikes_in_trial_soma_global[i].size();
				num_HVCRA_bursts += spikes_in_trial_dend_global[i].size();
				
			}
			
			std::cout << "Number of silent HVC-RA = " << num_HVCRA_silent << "\n" << std::endl;
			
			std::cout << "Average number of bursts of HVC-RA = " << static_cast<double>(num_HVCRA_bursts) / static_cast<double>(N_RA) << "\n" << std::endl;
			std::cout << "Average number of spikes of HVC-RA = " << static_cast<double>(num_HVCRA_spikes) / static_cast<double>(N_RA) << "\n" << std::endl;
			
			std::cout << "Average burst frequency of HVC-RA = " << static_cast<double>(num_HVCRA_bursts) * 1000.0 / (trial_duration * static_cast<double>(N_RA)) << "\n" << std::endl;
			std::cout << "Average spike frequency of HVC-RA = " << static_cast<double>(num_HVCRA_spikes) * 1000.0 / (trial_duration * static_cast<double>(N_RA)) << "\n" << std::endl;
			
			// spikes of HVC-I neurons
			size_t num_HVCI_spikes = 0; // total number of spikes of HVC-I neurons during trial
			
			for (int i = 0; i < N_I; i++)
				num_HVCI_spikes += spikes_in_trial_interneuron_global[i].size();
			
			if (N_I > 0){	
				std::cout << "Average number of spikes of HVC-I = " << static_cast<double>(num_HVCI_spikes) / static_cast<double>(N_I) << "\n" << std::endl;
				std::cout << "Average spike frequency of HVC-I = " << static_cast<double>(num_HVCI_spikes) * 1000.0 / (trial_duration * static_cast<double>(N_I)) << "\n" << std::endl;
			}
		}

        if (MPI_rank == 0)
        {
            for (int j = 0; j < N_RA; j++)
            {
                num_dendritic_spikes_in_trials[j][i] = static_cast<int>(spikes_in_trial_dend_global[j].size());
                num_somatic_spikes_in_trials[j][i] = static_cast<int>(spikes_in_trial_soma_global[j].size());
            }

        

			this->write_dend_spike_times((outputDirectory + fileSimName + std::to_string(i) + "_dendSpikes.bin").c_str());
			this->write_soma_spike_times((outputDirectory + fileSimName + std::to_string(i) + "_somaSpikes.bin").c_str());
			
			this->write_interneuron_spike_times((outputDirectory + fileSimName + std::to_string(i) + "_interneuron_spikes.bin").c_str());					
			
			if (record_excitatory_current) this->write_array(max_excitatory_currents, (outputDirectory + fileSimName + std::to_string(i) + "_max_excitatory_currents.bin").c_str());
		}
		this->randomize_after_trial();	
	}
	
	// process dendritic spikes

	std::vector<double> mean_burst_onset_time; // average of dendritic spike time
	std::vector<double> std_burst_onset_time; // standard deviation of dendritic spike time
    std::vector<double> average_num_dend_spikes_in_trial; // average number of dendritic spikes in trial
    std::vector<double> average_num_soma_spikes_in_trial; // average number of somatic spikes in trials

    std::vector<int> num_trials_with_bursts; // number of trials in which neuron produced bursts

	mean_burst_onset_time.resize(N_RA);
	std_burst_onset_time.resize(N_RA);
    average_num_dend_spikes_in_trial.resize(N_RA);
    average_num_soma_spikes_in_trial.resize(N_RA);
    num_trials_with_bursts.resize(N_RA);

	if (MPI_rank == 0)
	{
		for (int i = 0; i < N_RA; i++)
		{
            average_num_dend_spikes_in_trial[i] = std::accumulate(num_dendritic_spikes_in_trials[i].begin(), num_dendritic_spikes_in_trials[i].end(), 0.0) 
                                                / static_cast<double>(num_trials);

            average_num_soma_spikes_in_trial[i] = std::accumulate(num_somatic_spikes_in_trials[i].begin(), num_somatic_spikes_in_trials[i].end(), 0.0) 
                                                / static_cast<double>(num_trials);
            
            num_trials_with_bursts[i] = static_cast<int>(average_burst_onset_time[i].size());
			
          

            if (num_trials_with_bursts[i] > 0)
            {

                //for (int j = 0; j < (int) average_dendritic_spike_time[i].size(); j++)
                  //  printf("average_dendritic_spike_time[%d][%d] = %f\n", i, j, average_dendritic_spike_time[i][j]);
				mean_burst_onset_time[i] = std::accumulate(average_burst_onset_time[i].begin(), average_burst_onset_time[i].end(), 0.0) / 
                                        static_cast<double>(num_trials_with_bursts[i]);

            }

			else
				mean_burst_onset_time[i] = -1;
			// calculate standard deviation of burst times

			double accum = 0;
			double mean = mean_burst_onset_time[i];

			std::for_each(average_burst_onset_time[i].begin(), average_burst_onset_time[i].end(), [&accum, mean](const double t)
			{
				accum += (t - mean) * (t - mean);
			});

			if (static_cast<int>(average_burst_onset_time[i].size() > 1))
				std_burst_onset_time[i] = sqrt(accum / (static_cast<double>(average_burst_onset_time[i].size()) - 1));
			else
				std_burst_onset_time[i] = -1;


		}
	}

	std::string fileChainTest = outputDirectory + fileSimName + "manyTrials.bin";

	this->write_network_test(num_trials, num_trials_with_bursts, average_num_dend_spikes_in_trial, average_num_soma_spikes_in_trial, 
                           mean_burst_onset_time, std_burst_onset_time, fileChainTest.c_str());                          
}

std::vector<double> HvcNetwork::trial_distributed_training(double trial_duration, double spread, bool record_excitatory_current)
{
	// training neurons innervation
	std::vector<bool> indicators_training(N_RA);
	std::fill(indicators_training.begin(), indicators_training.end(), false);
	
	for (int i = 0; i < N_TR; i++)
		indicators_training[training_neurons[i]] = true;
		
	std::vector<bool> indicators_current_injected(N_RA);
	std::fill(indicators_current_injected.begin(), indicators_current_injected.end(), false);
	
	// array with times when training neurons are kicked
	std::vector<double> training_kick_times(N_RA);
	
	// array with max excitatory currents
	std::vector<double> max_excitatory_current_global;
	std::vector<double> max_excitatory_current_local;
	
	if (record_excitatory_current)
	{
		max_excitatory_current_global.resize(N_RA);
		max_excitatory_current_local.resize(N_RA_local);
		
		std::fill(max_excitatory_current_local.begin(), max_excitatory_current_local.end(), -1e6);
	}
	
	
	// update arrays for conductances
	std::vector<double> update_Ge_RA_local(N_RA);
	std::vector<double> update_Gi_RA_local(N_RA);
	std::vector<double> update_Ge_RA_global(N_RA);
	std::vector<double> update_Gi_RA_global(N_RA);
	
	std::vector<double> update_Ge_I_local(N_I);
	std::vector<double> update_Ge_I_global(N_I);
	
	// delivery queues
	std::vector<std::vector<std::pair<double,int>>> delivery_queue_RA_RA_soma(N_RA_local);
	std::vector<std::vector<std::pair<double,int>>> delivery_queue_RA_RA_dend(N_RA_local);
	std::vector<std::vector<std::pair<double,int>>> delivery_queue_RA_I(N_RA_local);
	
	std::vector<std::vector<std::pair<double,int>>> delivery_queue_I_RA(N_I_local);
	
	
	int some_RA_to_RA_soma_spike_delivered_local;
	int some_RA_to_RA_soma_spike_delivered_global;
	
	int some_RA_to_I_spike_delivered_local;
	int some_RA_to_I_spike_delivered_global;
	
	int some_I_to_RA_spike_delivered_local;
	int some_I_to_RA_spike_delivered_global;
    
    
    // generate kick times for training neurons
    if (MPI_rank == 0)
    {
		std::fill(training_kick_times.begin(), training_kick_times.end(), -1.0);
		
		for (int i = 0; i < N_TR; i++)
		{
			training_kick_times[training_neurons[i]] = noise_generator.random(spread) + WAITING_TIME - spread/2.0;
			//std::cout << "kick_times[" << training_neurons[i] << "] = " << training_kick_times[training_neurons[i]] << "\n";	
		}
		//std::cout << std::endl;
		
	}
	
	MPI_Bcast(&training_kick_times[0], N_RA, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double next_update_time = NETWORK_UPDATE_FREQUENCY;
    double network_time = 0;

	// initialize update arrays and fired indicators
	std::fill(update_Ge_RA_local.begin(), update_Ge_RA_local.end(), 0.0);
	std::fill(update_Ge_RA_global.begin(), update_Ge_RA_global.end(), 0.0);
	std::fill(update_Gi_RA_local.begin(), update_Gi_RA_local.end(), 0.0);
	std::fill(update_Gi_RA_global.begin(), update_Gi_RA_global.end(), 0.0);
	std::fill(update_Ge_I_local.begin(), update_Ge_I_local.end(), 0.0);
	std::fill(update_Ge_I_global.begin(), update_Ge_I_global.end(), 0.0);

    some_RA_to_RA_soma_spike_delivered_local = 0;
	some_RA_to_RA_soma_spike_delivered_global = 0;
	
	some_RA_to_I_spike_delivered_local = 0;
	some_RA_to_I_spike_delivered_global = 0;
	
	some_I_to_RA_spike_delivered_local = 0;
	some_I_to_RA_spike_delivered_global = 0;
	
	bool training_excited = false; // indicator if conductance pulse was delivered to training neurons
	
    // evolve dynamics
    size_t trial_size = static_cast<size_t>(round(trial_duration/TIMESTEP)) + 1;
    
    for (size_t t = 1; t < trial_size; t++)
	{
		network_time += TIMESTEP;
		
		for (int i = 0; i < N_RA_local; i++)
		{
			
			if ( ( indicators_training[Id_RA_local[i]] ) && ( !indicators_current_injected[Id_RA_local[i]] ) )
			{
				if ( network_time > training_kick_times[Id_RA_local[i]] )
				{
					HVCRA_local[i].raiseE(CONDUCTANCE_PULSE);
					indicators_current_injected[Id_RA_local[i]] = true;
				}
			}
			 
            // Debraband step
            HVCRA_local[i].Debraband_step_no_target_update();
            
            // in case excitatory current is recorded, update the array
            if (record_excitatory_current)
            {
				double Iexc = HVCRA_local[i].get_Iexc();
				
				if (Iexc > max_excitatory_current_local[i]) max_excitatory_current_local[i] = Iexc;
			}
            
		
			// check if somatic spike was delivered to some interneuron
			// loop through the delivery queue to check if current time exceeds the spike delivery time
			std::vector<std::pair<double,int>>::iterator it = delivery_queue_RA_I[i].begin();
		
			for (; it != delivery_queue_RA_I[i].end(); it++)
			{
				if (network_time >= it->first)
				{
				
					some_RA_to_I_spike_delivered_local = 1;
					int pos_in_local_target_array = it->second;
					int target_id = syn_ID_RA_I_local[i][pos_in_local_target_array];
				
					update_Ge_I_local[target_id] += weights_RA_I_local[i][pos_in_local_target_array];
				
					//~ std::cout << "HVC(RA) neuron " << Id_RA_local[i] << " at time " << internal_time 
							  //~ << " delivered spike with delivery time " << it->first 
							  //~ << " to HVC(I) neuron " << syn_ID_RA_I_local[i][pos_in_local_target_array] << std::endl;
				}
				else
					break;
			}
		
			// delete all processed spikes
			delivery_queue_RA_I[i].erase(delivery_queue_RA_I[i].begin(), it);
	   

			// check if somatic spike was delivered to some HVC(RA) neuron
			// loop through the delivery queue to check if current time exceeds the spike delivery time
            it = delivery_queue_RA_RA_soma[i].begin();
            
            for (; it != delivery_queue_RA_RA_soma[i].end(); it++)
            {
				if (network_time >= it->first)
				{
					some_RA_to_RA_soma_spike_delivered_local = 1;
					int pos_in_local_target_array = it->second;
					int target_id = syn_ID_RA_RA_local[i][pos_in_local_target_array];
					
					
					update_Ge_RA_local[target_id] += weights_RA_RA_local[i][pos_in_local_target_array];
					
					
					//std::cout << "Delivered spike from neuron " << Id_RA_local[i] << " to neuron " << target_id 
					//		  << " at time " << delivered_times_E2E_local[i][target_id].back() << std::endl;
				}
				else
					break;
			}
			
			// delete all processed spikes
			delivery_queue_RA_RA_soma[i].erase(delivery_queue_RA_RA_soma[i].begin(), it);
			
            // if some neuron produced somatic spike
            if (HVCRA_local[i].get_fired_soma())
            {
                spikes_in_trial_soma_local[i].push_back(network_time);
                
				// update delivery queues
				
			   
				// for inhibitory neurons
				// loop over all inhibitory targets of fired neurons
				size_t num_I_targets = syn_ID_RA_I_local[i].size();
			
				for (size_t j = 0; j < num_I_targets; j++)
				{
					double delivery_time = network_time + axonal_delays_RA_I_local[i][j];
				
					// if queue is empty, just add item to the queue
					if ( delivery_queue_RA_I[i].empty() )
						delivery_queue_RA_I[i].push_back(std::pair<double,int>(delivery_time, j));	
					// otherwise add item so that queue is sorted
					else
					{
						auto it = std::upper_bound(delivery_queue_RA_I[i].begin(), delivery_queue_RA_I[i].end(), std::pair<double,int>(delivery_time, j));
						delivery_queue_RA_I[i].insert(it, std::pair<double,int>(delivery_time, j));
					}
				}
               
					//std::cout << "neuron " << Id_RA_local[i] << " spike time = " << internal_time << " axonal delay = " << axonal_delays_RA_I[Id_RA_local[i]][syn_ID] << " delivery time RA to I: " << delivery_queue_RA_I[i].back().first << " delivery target id: " << syn_ID_RA_I_local[i][delivery_queue_RA_I[i].back().second] << std::endl;
				
				
				// loop over all excitatory targets
				size_t num_RA_targets = syn_ID_RA_RA_local[i].size();
				
				for (size_t j = 0; j < num_RA_targets; j++)
				{
					double delivery_time = network_time + axonal_delays_RA_RA_local[i][j];
					
					// if queue is empty, just add item to the queue
					if ( delivery_queue_RA_RA_soma[i].empty() )
						delivery_queue_RA_RA_soma[i].push_back(std::pair<double,int>(delivery_time, j));	
					// otherwise add item so that queue is sorted
					else
					{
						auto it = std::upper_bound(delivery_queue_RA_RA_soma[i].begin(), delivery_queue_RA_RA_soma[i].end(), std::pair<double,int>(delivery_time, j));
						delivery_queue_RA_RA_soma[i].insert(it, std::pair<double,int>(delivery_time, j));
					}
				}
			
				// sort delivery times in ascending order
				//std::sort(delivery_queue_RA_I[i].begin(), delivery_queue_RA_I[i].end());
				//std::sort(delivery_queue_RA_RA_soma[i].begin(), delivery_queue_RA_RA_soma[i].end());
                
            } 

            if (HVCRA_local[i].get_fired_dend())
            {
                spikes_in_trial_dend_local[i].push_back(network_time);
            }
		}

	
		for (int i = 0; i < N_I_local; i++)
		{
			HVCI_local[i].DP8_step_no_target_update();
		
			// check if somatic spike was delivered to some interneuron
			// loop through the delivery queue to check if current time exceeds the spike delivery time
			std::vector<std::pair<double,int>>::iterator it = delivery_queue_I_RA[i].begin();
			
			for (; it != delivery_queue_I_RA[i].end(); it++)
			{
				if (network_time >= it->first)
				{
					some_I_to_RA_spike_delivered_local = 1;
					int pos_in_local_target_array = it->second;
					int target_id = syn_ID_I_RA_local[i][pos_in_local_target_array];
				
					update_Gi_RA_local[target_id] += weights_I_RA_local[i][pos_in_local_target_array];
				
					//~ std::cout << "HVC(I) neuron " << Id_I_local[i] << " at time " << internal_time 
							  //~ << " delivered spike with delivery time " << it->first 
							  //~ << " to neuron " << syn_ID_I_RA_local[i][pos_in_local_target_array] << std::endl;
				}
				else
					break;
			}
		
			// delete all processed spikes
			delivery_queue_I_RA[i].erase(delivery_queue_I_RA[i].begin(), it);
			
			//  if some I neuron spikes, change conductance update array
			if (HVCI_local[i].get_fired())
			{
				//printf("My rank = %d; I neuron %d fired; spike_time = %f\n", MPI_rank, Id_I_local[i], internal_time);
				spikes_in_trial_interneuron_local[i].push_back(network_time);

				size_t num_RA_targets = syn_ID_I_RA_local[i].size();
				// loop over all targets of fired neurons
				for (size_t j = 0; j < num_RA_targets; j++)
				{
					double delivery_time = network_time + axonal_delays_I_RA_local[i][j];
				
					// if queue is empty, just add item to the queue
					if ( delivery_queue_I_RA[i].empty() )
						delivery_queue_I_RA[i].push_back(std::pair<double,int>(delivery_time, j));	
					// otherwise add item so that queue is sorted
					else
					{
						auto it = std::upper_bound(delivery_queue_I_RA[i].begin(), delivery_queue_I_RA[i].end(), std::pair<double,int>(delivery_time, j));
						delivery_queue_I_RA[i].insert(it, std::pair<double,int>(delivery_time, j));
					}
				}
			
			
			// sort delivery times in ascending order
			//std::sort(delivery_queue_I_RA[i].begin(), delivery_queue_I_RA[i].end());
		   }
		}
	

        // if we need to update network state
        // get if any neurons fired in some process

        if (network_time > next_update_time)
        {
			// gather all delivery indicators
			MPI_Allreduce(&some_RA_to_RA_soma_spike_delivered_local, &some_RA_to_RA_soma_spike_delivered_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&some_RA_to_I_spike_delivered_local, &some_RA_to_I_spike_delivered_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&some_I_to_RA_spike_delivered_local, &some_I_to_RA_spike_delivered_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            
           // if some interneuron spike was delivered to HVC(RA) neuron
            if (some_I_to_RA_spike_delivered_global > 0)
            {
            // sum update array and send to all processes

                MPI_Allreduce(&update_Gi_RA_local[0], &update_Gi_RA_global[0], N_RA, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                for (int i = 0; i < N_RA_local; i++)
                {
                    HVCRA_local[i].raiseI(update_Gi_RA_global[Id_RA_local[i]]);
                }
            	
				// update conductance arrays and delivered indicators
				some_I_to_RA_spike_delivered_global = 0;
				some_I_to_RA_spike_delivered_local = 0;
				
				std::fill(update_Gi_RA_local.begin(), update_Gi_RA_local.end(), 0.0);
				std::fill(update_Gi_RA_global.begin(), update_Gi_RA_global.end(), 0.0);
            }

            // if some somatic spike was delivered to interneuron update synaptic conductances
			if (some_RA_to_I_spike_delivered_global > 0)
			{
				// sum all update arrays and send to all processes

				MPI_Allreduce(&update_Ge_I_local[0], &update_Ge_I_global[0], N_I, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

				// now update excitatory conductances of all neurons

				for (int i = 0; i < N_I_local; i++)
				{
					HVCI_local[i].raiseE(update_Ge_I_global[Id_I_local[i]]);
				}

				// update conductance arrays and fired indicators
				some_RA_to_I_spike_delivered_local = 0;
				some_RA_to_I_spike_delivered_global = 0;

				std::fill(update_Ge_I_local.begin(), update_Ge_I_local.end(), 0.0);
				std::fill(update_Ge_I_global.begin(), update_Ge_I_global.end(), 0.0);
			}

			// if some somatic spike was delivered to HVC(RA) neuron update synaptic conductances
			if (some_RA_to_RA_soma_spike_delivered_global > 0)
			{
				// sum all update arrays and send to all processes

				MPI_Allreduce(&update_Ge_RA_local[0], &update_Ge_RA_global[0], N_RA, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

				// now update excitatory conductances of all neurons
				for (int i = 0; i < N_RA_local; i++)
				{
					HVCRA_local[i].raiseE(update_Ge_RA_global[Id_RA_local[i]]); // update conductance
				}

				// update conductance arrays and fired indicators
				some_RA_to_RA_soma_spike_delivered_global = 0;
				some_RA_to_RA_soma_spike_delivered_local = 0;
				
				std::fill(update_Ge_RA_local.begin(), update_Ge_RA_local.end(), 0.0);
				std::fill(update_Ge_RA_global.begin(), update_Ge_RA_global.end(), 0.0);
			}	

            next_update_time += NETWORK_UPDATE_FREQUENCY;
        } // end network update


        //MPI_Barrier(MPI_COMM_WORLD);
    }  
	// end evolve dynamics
	
	// gather recorded excitatory current
	if (record_excitatory_current) 
		this->gather_local_arrays(max_excitatory_current_local, max_excitatory_current_global);
		
	return max_excitatory_current_global;
            
}

void HvcNetwork::resize_arrays_for_I(int n_local, int n_total)
{
    HVCI_local.resize(n_local);

    // connections and their ids
	weights_I_RA_local.resize(n_local);
	syn_ID_I_RA_local.resize(n_local);
	
	// axonal delays
	axonal_delays_I_RA_local.resize(n_local);
    
    spikes_in_trial_interneuron_global.resize(n_total);
    spikes_in_trial_interneuron_local.resize(n_local);
}

void HvcNetwork::resize_global_arrays()
{
	// connections
	weights_RA_I.resize(N_RA);
	syn_ID_RA_I.resize(N_RA);
	
	weights_I_RA.resize(N_I);
	syn_ID_I_RA.resize(N_I);
	
	weights_RA_RA.resize(N_RA);
	syn_ID_RA_RA.resize(N_RA);
		
	// axonal delays
	axonal_delays_RA_I.resize(N_RA);
	axonal_delays_RA_RA.resize(N_RA);
	axonal_delays_I_RA.resize(N_I);	
}

void HvcNetwork::resize_arrays_for_RA(int n_local, int n_total)
{
	HVCRA_local.resize(n_local);
  
    // connections and their ids
    weights_RA_RA_local.resize(n_local);
    syn_ID_RA_RA_local.resize(n_local);
    
    weights_RA_I_local.resize(n_local);
    syn_ID_RA_I_local.resize(n_local);
    
    // axonal delays
    axonal_delays_RA_I_local.resize(n_local);
    axonal_delays_RA_RA_local.resize(n_local);
    
    spikes_in_trial_soma_global.resize(n_total);
    spikes_in_trial_dend_global.resize(n_total);

    spikes_in_trial_soma_local.resize(n_local);
    spikes_in_trial_dend_local.resize(n_local);
}

void HvcNetwork::gather_local_arrays(const std::vector<double> &local, std::vector<double> &global)
{
	int *recvcounts = new int[MPI_size];
    int *displs = new int[MPI_size];
	
	
	recvcounts[0] = N_RA_sizes[0];
	displs[0] = 0;

	for (int i = 1; i < MPI_size; i++)
	{
		recvcounts[i] = N_RA_sizes[i];
		displs[i] = displs[i-1] + recvcounts[i-1];
	}


    // all gatherv functions collect data from local processes. Data is a concatenated version of local 
    // data on the processes. 
	MPI_Allgatherv(&local[0], N_RA_local, MPI_DOUBLE,
        &global[0], recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

	delete [] recvcounts;
    delete [] displs;
}

void HvcNetwork::gather_data()
{
    MPI_Status status;
    // Gather all data to master process
   
    int *spike_num_soma_local = new int[N_RA_local];
    int *spike_num_soma_global = new int[N_RA];
    int *spike_num_dend_local = new int[N_RA_local];
    int *spike_num_dend_global = new int[N_RA];
    int *spike_num_interneuron_local = new int[N_I_local];
    int *spike_num_interneuron_global = new int[N_I];
    int *recvcounts_RA = new int[MPI_size];
    int *displs_RA = new int[MPI_size];
	int *recvcounts_I = new int[MPI_size];
    int *displs_I = new int[MPI_size];


    if (MPI_rank == 0)
    {
        recvcounts_RA[0] = N_RA_sizes[0];
        displs_RA[0] = 0;

        for (int i = 1; i < MPI_size; i++)
        {
            recvcounts_RA[i] = N_RA_sizes[i];
            displs_RA[i] = displs_RA[i-1] + recvcounts_RA[i-1];
        }
		
		recvcounts_I[0] = N_I_sizes[0];
        displs_I[0] = 0;

        for (int i = 1; i < MPI_size; i++)
        {
            recvcounts_I[i] = N_I_sizes[i];
            displs_I[i] = displs_I[i-1] + recvcounts_I[i-1];
        }

    }

    for (int i = 0; i < N_RA_local; i++)
    {
        spike_num_soma_local[i] = spikes_in_trial_soma_local[i].size();
        spike_num_dend_local[i] = spikes_in_trial_dend_local[i].size();

        //printf("Rank = %d, supersyn_sizes_local[%d] = %d\n", MPI_rank, Id_RA_local[i], supersyn_sizes_local[i]);
        //printf("Rank = %d, syn_sizes_local[%d] = %d\n", MPI_rank, Id_RA_local[i], syn_sizes_local[i]);
        //printf("Rank = %d, spike_num_soma_local[%d] = %d\n", MPI_rank, Id_RA_local[i], spike_num_soma_local[i]);
        //printf("Rank = %d, spike_num_dend_local[%d] = %d\n", MPI_rank, Id_RA_local[i], spike_num_dend_local[i]);
    }

    // all gatherv functions collect data from local processes. Data is a concatenated version of local 
    // data on the processes. Therefore neuronal ids may not be monotonic!!! Data is rearranged only 
    // writing to files

	for (int i = 0; i < N_I_local; i++)
		spike_num_interneuron_local[i] = (int) spikes_in_trial_interneuron_local[i].size();

	
	MPI_Gatherv(&spike_num_interneuron_local[0], N_I_local, MPI_INT,
        &spike_num_interneuron_global[0], recvcounts_I, displs_I, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(&spike_num_soma_local[0], N_RA_local, MPI_INT,
        &spike_num_soma_global[0], recvcounts_RA, displs_RA, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(&spike_num_dend_local[0], N_RA_local, MPI_INT,
        &spike_num_dend_global[0], recvcounts_RA, displs_RA, MPI_INT, 0, MPI_COMM_WORLD);
	
    // Receive functions on the other hand rearrange data immediately. Thus, there is no need
    // to take special care while writing to files

    if (MPI_rank == 0)
    {
        for (int i = 0; i < N_RA; i++)
        {
            spikes_in_trial_soma_global[i].resize(spike_num_soma_global[i]);
            spikes_in_trial_dend_global[i].resize(spike_num_dend_global[i]);
        }


        // Copy from master's local arrays to master's global arrays
        for (int i = 0; i < N_RA_local; i++)
        {
            spikes_in_trial_soma_global[i] = spikes_in_trial_soma_local[i];
            spikes_in_trial_dend_global[i] = spikes_in_trial_dend_local[i];
        }

    // Receive from others
		int N = N_RA_sizes[0]; // number of RA neurons in the processes with lower rank

        for (int i = 1; i < MPI_size; i++)
        {

            for (int j = 0; j < N_RA_sizes[i]; j++)
            {
                int count;
				int receive_index = N + j;
                
                if (spike_num_soma_global[receive_index] != 0)
                {
                    MPI_Recv(&spikes_in_trial_soma_global[receive_index][0],
                        spike_num_soma_global[receive_index], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);

                    MPI_Get_count(&status, MPI_INT, &count);
                    //printf("Recv spikes in trial; from i = %d  count = %d\n", i, count);
                }

                if (spike_num_dend_global[receive_index] != 0)
                {
                    MPI_Recv(&spikes_in_trial_dend_global[receive_index][0],
                        spike_num_dend_global[receive_index], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);

                    MPI_Get_count(&status, MPI_INT, &count);
                    //printf("Recv spikes in trial; from i = %d  count = %d\n", i, count);
                }
            }

			N += N_RA_sizes[i];

        }

    }

    else
    {
        for (int i = 0; i < N_RA_local; i++)
        {
         
            if (spike_num_soma_local[i] != 0)
                MPI_Send(&spikes_in_trial_soma_local[i][0],
                        spike_num_soma_local[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

            if (spike_num_dend_local[i] != 0)
                MPI_Send(&spikes_in_trial_dend_local[i][0],
                        spike_num_dend_local[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

	// gather spikes of interneurons
	if (MPI_rank == 0)
    {
    
        for (int i = 0; i < N_I; i++)
			spikes_in_trial_interneuron_global[i].resize(spike_num_interneuron_global[i]);
        
	    for (int i = 0; i < N_I_local; i++)
			spikes_in_trial_interneuron_global[i] = spikes_in_trial_interneuron_local[i];
	
        int N = N_I_sizes[0]; // number of I neurons in the processes with lower rank

        for (int i = 1; i < MPI_size; i++)
        {
            for (int j = 0; j < N_I_sizes[i]; j++)
            {
                int count;
				int receive_index = N + j;

                if (spike_num_interneuron_global[receive_index] != 0)
                {
                    MPI_Recv(&spikes_in_trial_interneuron_global[receive_index][0],
                        spike_num_interneuron_global[receive_index], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);

                    MPI_Get_count(&status, MPI_INT, &count);
                }


            }

			N += N_I_sizes[i];
        }
    }

    
    else
    {
        for (int i = 0; i < N_I_local; i++)
        {
            if (spike_num_interneuron_local[i] != 0)
                MPI_Send(&spikes_in_trial_interneuron_local[i][0],
                    spike_num_interneuron_local[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        }
    }
    
    delete [] recvcounts_RA;
    delete [] displs_RA;
    delete [] recvcounts_I;
    delete [] displs_I;
    delete [] spike_num_soma_local;
    delete [] spike_num_soma_global;
    delete [] spike_num_dend_local;
    delete [] spike_num_dend_global;
	delete [] spike_num_interneuron_local;
	delete [] spike_num_interneuron_global;
}

void HvcNetwork::get_neuronRA_location(int n, int* rank, int* shift)
{
	int i = 0;
	int N = 0;

	while (n >= N)
	{
		if (n < N + N_RA_sizes[i])
		{
			*rank = i;
			*shift = n - N;
			
			//if (MPI_rank == 0)
				//std::cout << "id = " << n << " rank = " << *rank << " shift = " << *shift << std::endl;
			break;
		}
		
		N += N_RA_sizes[i];
		i++;
	}
}

void HvcNetwork::get_neuronI_location(int n, int* rank, int* shift)
{
	int i = 0;
	int N = 0;

	while (n >= N)
	{
		if (n < N + N_I_sizes[i])
		{
			*rank = i;
			*shift = n - N;
			
			break;
		}
		
		N += N_I_sizes[i];
		i++;
	}
}

void HvcNetwork::write_experimental_network_to_directory(std::string outputDirectory)
{
	std::string filename_connections_RA2I = outputDirectory + "RA_I_connections.bin";
	std::string filename_connections_I2RA = outputDirectory + "I_RA_connections.bin";
	std::string filename_connections_RA2RA = outputDirectory + "RA_RA_connections.bin";
	
	std::string filename_parameters = outputDirectory + "parameters.bin";
	
	this->write_synapses(syn_ID_RA_RA, weights_RA_RA, axonal_delays_RA_RA, filename_connections_RA2RA.c_str());
	this->write_synapses(syn_ID_RA_I,  weights_RA_I,  axonal_delays_RA_I,  filename_connections_RA2I.c_str());
	this->write_synapses(syn_ID_I_RA,  weights_I_RA,  axonal_delays_I_RA,  filename_connections_I2RA.c_str());
	
    this->write_parameters_to_file(filename_parameters.c_str());
}

void HvcNetwork::write_noise_based_on_dend_capacitance(const std::vector<double>& mu_soma, const std::vector<double>& std_soma, 
													   const std::vector<double>& mu_dend, const std::vector<double>& std_dend,
													   const char* filename){
	std::ofstream out;
	
	// open files
	out.open(filename, std::ios::binary | std::ios::out);
	
	assert(N_RA == static_cast<int>(mu_soma.size()));
	assert(N_RA == static_cast<int>(std_soma.size()));
	assert(N_RA == static_cast<int>(mu_dend.size()));
	assert(N_RA == static_cast<int>(std_dend.size()));
	
	// write number of neurons
	out.write(reinterpret_cast<char *>(&N_RA), sizeof(N_RA));
	
	for (int i = 0; i < N_RA; i++){
		out.write(reinterpret_cast<const char *>(&mu_soma[i]), sizeof(double));
		out.write(reinterpret_cast<const char *>(&std_soma[i]), sizeof(double));
		out.write(reinterpret_cast<const char *>(&mu_dend[i]), sizeof(double));
		out.write(reinterpret_cast<const char *>(&std_dend[i]), sizeof(double));
	}
	
	out.close();													
}

void HvcNetwork::write_capacitance_and_integration_time(const std::vector<double>& c, 
										const std::vector<double>& it, const char* filename){
	std::ofstream out;
	
	// open files
	out.open(filename, std::ios::binary | std::ios::out);
	
	assert(N_RA == static_cast<int>(c.size()));
	assert(N_RA == static_cast<int>(it.size()));
	
	// write number of neurons
	out.write(reinterpret_cast<char *>(&N_RA), sizeof(N_RA));
	
	for (int i = 0; i < N_RA; i++){
		out.write(reinterpret_cast<const char *>(&c[i]), sizeof(double));
		out.write(reinterpret_cast<const char *>(&it[i]), sizeof(double));
	}
	out.close();
}

void HvcNetwork::write_synapses(const std::vector<std::vector<int>>& syn_ID,
								const std::vector<std::vector<double>>& weights,
								const std::vector<std::vector<double>>& delays,
								const char* filename)
{
	std::ofstream out;
	
	// open files
	out.open(filename, std::ios::binary | std::ios::out);
	
	
	int N = static_cast<int>(syn_ID.size()); // number of source neurons
	
	// write number of neurons
	out.write(reinterpret_cast<char *>(&N), sizeof(N));
	
	for (int i = 0; i < N; i++)
	{
		int num_targets = static_cast<int>(syn_ID[i].size());

		out.write(reinterpret_cast<char *>(&i), sizeof(i));
		out.write(reinterpret_cast<char *>(&num_targets), sizeof(num_targets));

		for (int j = 0; j < num_targets; j++)
		{
			out.write(reinterpret_cast<const char *>(&syn_ID[i][j]), sizeof(int));
			out.write(reinterpret_cast<const char *>(&weights[i][j]), sizeof(double));
			out.write(reinterpret_cast<const char *>(&delays[i][j]), sizeof(double));
			
		}

	}
	out.close();
}

void HvcNetwork::write_burst_labels(const std::vector<double>& labels, const char* filename)
{
	std::ofstream out;
  
	out.open(filename, std::ios::out | std::ios::binary );
	
	int number_of_neurons = static_cast<int>(labels.size());
	
	out.write(reinterpret_cast<char *>(&number_of_neurons), sizeof(number_of_neurons));
	
	for (int i = 0; i < number_of_neurons; i++)
		out.write(reinterpret_cast<const char *>(&labels[i]), sizeof(labels[i]));
	
	out.close();
}

void HvcNetwork::write_parameters_to_file(const char* filename)
{
	std::ofstream out;
  
	out.open(filename, std::ios::out | std::ios::binary );
	
	out.write(reinterpret_cast<char *>(&N_RA), sizeof(N_RA));
	out.write(reinterpret_cast<char *>(&N_I), sizeof(N_I));
	
	out.write(reinterpret_cast<char *>(&Gie_max), sizeof(Gie_max));
	out.write(reinterpret_cast<const char *>(&Gei_max), sizeof(Gei_max));
	out.write(reinterpret_cast<char *>(&Gee_max), sizeof(Gee_max));
	
	out.close();
}

void HvcNetwork::write_network_test(int num_trials, std::vector<int>& num_trials_with_dend_spikes, std::vector<double>& average_num_dend_spikes_in_trials, 
                                    std::vector<double>& average_num_soma_spikes_in_trials, std::vector<double>& mean_burst_time, 
                                    std::vector<double>& std_burst_time, const char* filename)
{
    if (MPI_rank == 0)
    {
        std::ofstream out;

        out.open(filename, std::ios::out | std::ios::binary);

        // write number of neurons to each file

        out.write(reinterpret_cast<char *>(&N_RA), sizeof(N_RA));
        out.write(reinterpret_cast<char *>(&num_trials), sizeof(num_trials));

        for (int i = 0; i < N_RA; i++)
        {
            double firing_robustness = num_trials_with_dend_spikes[i] / static_cast<double>(num_trials);

            out.write(reinterpret_cast<char *>(&firing_robustness), sizeof(firing_robustness));
            out.write(reinterpret_cast<char *>(&average_num_dend_spikes_in_trials[i]), sizeof(average_num_dend_spikes_in_trials[i]));
            out.write(reinterpret_cast<char *>(&average_num_soma_spikes_in_trials[i]), sizeof(average_num_soma_spikes_in_trials[i]));
            out.write(reinterpret_cast<char *>(&mean_burst_time[i]), sizeof(mean_burst_time[i]));
            out.write(reinterpret_cast<char *>(&std_burst_time[i]), sizeof(std_burst_time[i]));
        }
        out.close();
    }
}

void HvcNetwork::write_array(std::vector<double> a, const char* filename)
{
	int N = static_cast<int>(a.size()); // array size
	
	std::ofstream out;
	out.open(filename, std::ios::out | std::ios::binary );
	out.write(reinterpret_cast<char *>(&N), sizeof(N));

	for (double x : a)
		out.write(reinterpret_cast<char *>(&x), sizeof(double));

	out.close();	
}

void HvcNetwork::write_soma_spike_times(const char* filename)
{
    if (MPI_rank == 0)
    {
        std::ofstream out;

        out.open(filename, std::ios::out | std::ios::binary );
        out.write(reinterpret_cast<char *>(&N_RA), sizeof(N_RA));

        //for (int i = 0; i < N_RA; i++)
        //    printf("spike_times[%d] = %f\n", i, spike_times[i]);
        // write spike times
        for (int i = 0; i < N_RA; i++)
        {
            int spike_array_size = spikes_in_trial_soma_global[i].size();
            //printf("Neuron %d; number of somatic spikes in trial: %d\n", i, spike_array_size);
            out.write(reinterpret_cast<char *>(&spike_array_size), sizeof(int));

            for (int j = 0; j < spike_array_size; j++)
	        {
       		//printf("relative_spike_time_soma = %f\n", relative_spike_time);
                out.write(reinterpret_cast<char *>(&spikes_in_trial_soma_global[i][j]), sizeof(spikes_in_trial_soma_global[i][j]));
	        }
        }
        //out.write(reinterpret_cast<char *>(spike_times), N_RA*sizeof(double));

        out.close();
    }
}

void HvcNetwork::write_dend_spike_times(const char* filename)
{
    if (MPI_rank == 0)
    {
        std::ofstream out;

        out.open(filename, std::ios::out | std::ios::binary );
        out.write(reinterpret_cast<char *>(&N_RA), sizeof(N_RA));

        //for (int i = 0; i < N_RA; i++)
        //    printf("spike_times[%d] = %f\n", i, spike_times[i]);
        // write spike times
        for (int i = 0; i < N_RA; i++)
        {
            int spike_array_size = spikes_in_trial_dend_global[i].size();
            //printf("Neuron %d; number of dendritic spikes in trial: %d\n", i, spike_array_size);
            out.write(reinterpret_cast<char *>(&spike_array_size), sizeof(int));
	    
            for (int j = 0; j < spike_array_size; j++)
			{
                //out.write(reinterpret_cast<char *>(&spikes_in_trial_dend_global[i][j]), sizeof(double));
        	out.write(reinterpret_cast<char *>(&spikes_in_trial_dend_global[i][j]), sizeof(spikes_in_trial_dend_global[i][j]));
			//printf("Neuron %d; relative spike time = %f\n", i, relative_spike_time);
			}
		}
        //out.write(reinterpret_cast<char *>(spike_times), N_RA*sizeof(double));

        out.close();
    }
}

void HvcNetwork::write_interneuron_spike_times(const char* filename)
{
    if (MPI_rank == 0)
    {
        std::ofstream out;

        out.open(filename, std::ios::out | std::ios::binary );
        out.write(reinterpret_cast<char *>(&N_I), sizeof(N_I));

        //for (int i = 0; i < N_RA; i++)
        //    printf("spike_times[%d] = %f\n", i, spike_times[i]);
        // write spike times
        for (int i = 0; i < N_I; i++)
        {
            int spike_array_size = spikes_in_trial_interneuron_global[i].size();
            //printf("Neuron %d; number of dendritic spikes in trial: %d\n", i, spike_array_size);
            out.write(reinterpret_cast<char *>(&spike_array_size), sizeof(int));
	    
            for (int j = 0; j < spike_array_size; j++)
	    {
                //out.write(reinterpret_cast<char *>(&spikes_in_trial_dend_global[i][j]), sizeof(double));
        	out.write(reinterpret_cast<char *>(&spikes_in_trial_interneuron_global[i][j]), sizeof(spikes_in_trial_interneuron_global[i][j]));
			//printf("Neuron %d; relative spike time = %f\n", i, relative_spike_time);
	    }
	}
        //out.write(reinterpret_cast<char *>(spike_times), N_RA*sizeof(double));

        out.close();
    }
}

void HvcNetwork::read_number_of_neurons(const char* filename)
{
	std::ifstream inp;

	// open files
	inp.open(filename, std::ios::binary | std::ios::in);
	
	// read number of neurons	
	inp.read(reinterpret_cast<char*>(&N_RA), sizeof(N_RA));
	inp.read(reinterpret_cast<char*>(&N_I), sizeof(N_I));
	
	//std::cout << "Number of HVC(RA) neurons from parameter file = = " << N_RA << std::endl;
	//std::cout << "Number of HVC(I) neurons from parameter file = " << N_I << std::endl;
	
	// close files
	inp.close();		
}

void HvcNetwork::prepare_network_for_polychronous_wiring(const std::string& fileTraining)
{	
	this->prepare_slaves_for_testing(fileTraining);
}

void HvcNetwork::prepare_network_for_testing(std::string networkDir, std::string fileTraining, bool resample)
{
	std::string fileParameters = networkDir + "parameters.bin";
	
	if (MPI_rank == 0)
	{
		this->read_number_of_neurons(fileParameters.c_str()); 
		
		std::string filename_connections_RA2I = networkDir + "RA_I_connections.bin";
		std::string filename_connections_I2RA = networkDir + "I_RA_connections.bin";
		std::string filename_connections_RA2RA = networkDir + "RA_RA_connections.bin";

		this->read_synapses(syn_ID_RA_RA, weights_RA_RA, axonal_delays_RA_RA, filename_connections_RA2RA.c_str());
		this->read_synapses(syn_ID_RA_I,  weights_RA_I,  axonal_delays_RA_I,  filename_connections_RA2I.c_str());
		this->read_synapses(syn_ID_I_RA,  weights_I_RA,  axonal_delays_I_RA,  filename_connections_I2RA.c_str());
		
		
		if (resample) this->resample_strength_of_connections_in_network();
	}
	
	this->prepare_slaves_for_testing(fileTraining);
}

void HvcNetwork::read_capacitance(const char* file_capacitance, const char* file_noise){
	std::vector<double> capacitance_global;
	std::vector<double> mu_soma_global;
	std::vector<double> std_soma_global;
	std::vector<double> mu_dend_global;
	std::vector<double> std_dend_global;
	
	if (MPI_rank == 0){
		std::ifstream inp;
			
		// open files
		inp.open(file_capacitance, std::ios::binary | std::ios::in);
		
		int N;
		
		inp.read(reinterpret_cast<char *>(&N), sizeof(N));
		
		assert(N == N_RA);
		
		capacitance_global.resize(N_RA);
		double tmp;
		
		for (int i = 0; i < N_RA; i++){
			inp.read(reinterpret_cast<char *>(&capacitance_global[i]), sizeof(double));
			inp.read(reinterpret_cast<char *>(&tmp), sizeof(double));
			
		}
		inp.close();
		
		// sample noise based on dendritic capacitance
		this->sample_noise_based_on_dend_capacitance(capacitance_global,
				mu_soma_global, std_soma_global, 
				mu_dend_global, std_dend_global);
		
		this->write_noise_based_on_dend_capacitance(mu_soma_global, std_soma_global, 
													mu_dend_global, std_dend_global,
													file_noise);
	}
	std::vector<double> capacitance_local;
	std::vector<double> mu_soma_local;
	std::vector<double> std_soma_local;
	std::vector<double> mu_dend_local;
	std::vector<double> std_dend_local;
	
	this->scatter_global_to_local_double(capacitance_global, capacitance_local);
	this->scatter_global_to_local_double(mu_soma_global, mu_soma_local);
	this->scatter_global_to_local_double(std_soma_global, std_soma_local);
	this->scatter_global_to_local_double(mu_dend_global, mu_dend_local);
	this->scatter_global_to_local_double(std_dend_global, std_dend_local);
	
	for (int i = 0; i < N_RA_local; i++){
		HVCRA_local[i].set_cm_dend(capacitance_local[i]);
		HVCRA_local[i].set_white_noise(mu_soma_local[i], std_soma_local[i], 
									   mu_dend_local[i], std_dend_local[i]);
	}
		
	//this->set_noise_based_on_dend_capacitance(capacitance_local);
}

void HvcNetwork::read_synapses(std::vector<std::vector<int>>& syn_ID,
							   std::vector<std::vector<double>>& weights,
							   std::vector<std::vector<double>>& axonal_delays,
							   const char* filename)
{
	std::ifstream inp;
        
	// open files
	inp.open(filename, std::ios::binary | std::ios::in);
	
	int N;
	
	inp.read(reinterpret_cast<char *>(&N), sizeof(N));
	
	//std::cout << "Number of neurons read from file " << filename << " N = " << N << std::endl;
	
	// resize connections
	weights.resize(N);
	syn_ID.resize(N);
	axonal_delays.resize(N);
	
	// read connections
	for (int i = 0; i < N; i++)
	{
		int n_id; // neuronal id
		int num_targets; // number of outgoing connections

		inp.read(reinterpret_cast<char *>(&n_id), sizeof(n_id));
		inp.read(reinterpret_cast<char *>(&num_targets), sizeof(num_targets)); // write neuron's ID

		syn_ID[i].resize(num_targets);
		weights[i].resize(num_targets);
		axonal_delays[i].resize(num_targets);
		
		for (int j = 0; j < num_targets; j++)
		{
			inp.read(reinterpret_cast<char *>(&syn_ID[i][j]), sizeof(int));
			inp.read(reinterpret_cast<char *>(&weights[i][j]), sizeof(double));
			inp.read(reinterpret_cast<char *>(&axonal_delays[i][j]), sizeof(double));

		}

	}
	
	inp.close();
}

void HvcNetwork::read_training_neurons(const char* filename)
{
	std::ifstream inp;
  
	inp.open(filename, std::ios::in | std::ios::binary);

	inp.read(reinterpret_cast<char *>(&N_TR), sizeof(N_TR));
	
	//std::cout << "Number of training neurons read from file with training neurons N_TR = " << N_TR << std::endl; 
				 
	training_neurons.resize(N_TR);
	
	for (size_t i = 0; i < N_TR; i++)
		inp.read(reinterpret_cast<char *>(&training_neurons[i]), sizeof(training_neurons[i]));
	
	inp.close();
}

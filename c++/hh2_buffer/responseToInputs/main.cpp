#include "HH2.h"
#include "poisson_noise.h"
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

const double TRIAL_DURATION = 100.0; // trial duration in ms
const double TIMESTEP = 0.005; // time resolution of dynamics
const unsigned SEED = 1991; // seed for random number generator
const double TRAINING_KICK = 3.0; // excitatory conductance kick delivered to training neurons
	
const double white_noise_mean_soma = 0.0;
const double white_noise_std_soma = 0.0;
const double white_noise_mean_dend = 0.0;
const double white_noise_std_dend = 0.0;

void write_results_to_file(const std::vector<double>& burst_onset_times, const std::vector<int>& num_spikes_in_trial,
							std::string filename){
	std::ofstream output;

	output.open(filename, std::ios::out | std::ios::binary); 
	
	int num_trials = static_cast<int>(num_spikes_in_trial.size());
	int num_trials_with_burst = static_cast<int>(burst_onset_times.size());
	
	output.write(reinterpret_cast<const char*>(&num_trials), sizeof(int));
	for (size_t i = 0; i < num_spikes_in_trial.size(); i++)
		output.write(reinterpret_cast<const char*>(&num_spikes_in_trial[i]), sizeof(int));
	
	output.write(reinterpret_cast<const char*>(&num_trials_with_burst), sizeof(int));	
	for (size_t i = 0; i < burst_onset_times.size(); i++)
		output.write(reinterpret_cast<const char*>(&burst_onset_times[i]), sizeof(double));
	
	output.close();
}

std::vector<double> sample_input_times(int num_inputs, double mean_input_time, 
										double synch_window, Poisson_noise& noise_generator){
	std::vector<double> input_times(num_inputs);
	
	for (int i = 0; i < num_inputs; i++)
		input_times[i] = mean_input_time - synch_window/2. + noise_generator.random(synch_window); 
	
	return(input_times);
}

std::vector<double> sample_input_weights(int num_inputs, double gmax, Poisson_noise& noise_generator){
	std::vector<double> input_weights(num_inputs);
	
	for (int i = 0; i < num_inputs; i++)
		input_weights[i] = noise_generator.random(gmax);
		
	return(input_weights);
}

void simulate_input(int num_inputs, double gee_max, double mean_input_time, 
					double synch_window, int num_trial){				
	HH2 n;
	Poisson_noise noise_generator;
	
	noise_generator.set_seed(SEED); 
	n.set_noise_generator(&noise_generator);
	n.set_dynamics(TIMESTEP);
	
	
	n.set_white_noise(white_noise_mean_soma, white_noise_std_soma,
					  white_noise_mean_dend, white_noise_std_dend);
	
	std::vector<double> burst_onset_times;
	
	for (int t = 0; t < num_trial; t++){	
		std::vector<double> input_times = sample_input_times(num_inputs, mean_input_time, synch_window, noise_generator);
		std::sort(input_times.begin(), input_times.end());
		
		//std::cout << "input_times.size() = " << input_times.size() << std::endl;
		std::vector<int> ind_input_times(input_times.size());
		
		//std::cout << "input_times, ind_input_times:" <<  std::endl;
		for (size_t i = 0; i < input_times.size(); i++){
			ind_input_times[i] = static_cast<int>(input_times[i] / TIMESTEP);
			//std::cout << input_times[i] << ", " << ind_input_times[i] << " " << std::endl;
		}
		
		std::vector<double> input_weights = sample_input_weights(num_inputs, gee_max, noise_generator);
		
		//std::cout << "input_weights:" <<  std::endl;
		//for (size_t i = 0; i < input_weights.size(); i++)
		//	std::cout << input_weights[i] << " ";
		//std::cout << std::endl;
		
		bool all_inputs_delivered = false; // indicator that all inputs were delivered
		bool first_spike = true; // indicator that produce spike is the first one
		int next_input_ind = 0; // index of the next input to be delivered

		for (int i = 0; i < static_cast<int>(TRIAL_DURATION/TIMESTEP); i++){
			if (i >= ind_input_times.front()){
				while ((!all_inputs_delivered) && (i == ind_input_times[next_input_ind])){
					//std::cout << "Delivered input at time " << static_cast<double>(i) * timestep << std::endl;
					n.raiseE(input_weights[next_input_ind]);
					next_input_ind += 1;
					
					if (next_input_ind >= num_inputs)
						all_inputs_delivered = true;	
				}
				
			}
			
			n.Debraband_step_no_target_update();
			
			if (n.get_fired_soma())
			{
				double spike_time = static_cast<double>(i) * TIMESTEP;
				std::cout << "Spike at " << spike_time << std::endl;
				
				if (first_spike) {
					burst_onset_times.push_back(spike_time - mean_input_time);
					first_spike = false;
				}
			}
		}
		
		if (all_inputs_delivered)
			std::cout << "All inputs were delivered!" << std::endl;
		
		n.set_to_rest();
	}
	
	std::cout << "Burst onset times:\n"; 
	for (size_t i = 0; i < burst_onset_times.size(); i++)
		std::cout << burst_onset_times[i] << " ";
	std::cout << std::endl;						
}

void simulate_input_from_neurons(int num_inputs, double gee_max, double mean_kick_time, 
					double synch_window, int num_trial, double cm_d, std::string filename){				
	HH2 target_neuron;
	std::vector<HH2> source_neurons(num_inputs);
	
	Poisson_noise noise_generator;
	
	noise_generator.set_seed(SEED); 
	
	target_neuron.set_cm_dend(cm_d);
	target_neuron.set_noise_generator(&noise_generator);
	target_neuron.set_dynamics(TIMESTEP);
	target_neuron.set_white_noise(white_noise_mean_soma, white_noise_std_soma,
					  white_noise_mean_dend, white_noise_std_dend);
	
	for (int i = 0; i < num_inputs; i++){
		source_neurons[i].set_noise_generator(&noise_generator);
		source_neurons[i].set_dynamics(TIMESTEP);
	} 
	
	std::vector<double> burst_onset_times;
	std::vector<int> num_spikes_in_trial;
	
	for (int t = 0; t < num_trial; t++){	
		std::vector<double> kick_times = sample_input_times(num_inputs, mean_kick_time, synch_window, noise_generator);
		std::vector<double> input_times(num_inputs);
		
		//std::cout << "input_times.size() = " << input_times.size() << std::endl;
		std::vector<int> ind_kick_times(kick_times.size());
		
		//std::cout << "input_times, ind_input_times:" <<  std::endl;
		for (size_t i = 0; i < kick_times.size(); i++){
			ind_kick_times[i] = static_cast<int>(kick_times[i] / TIMESTEP);
			//std::cout << input_times[i] << ", " << ind_input_times[i] << " " << std::endl;
		}
		
		std::vector<double> input_weights = sample_input_weights(num_inputs, gee_max, noise_generator);
		
		//std::cout << "input_weights:" <<  std::endl;
		//for (size_t i = 0; i < input_weights.size(); i++)
		//	std::cout << input_weights[i] << " ";
		//std::cout << std::endl;
		
		bool first_spike_target = true; // indicator that produced spike of target neuron is the first one
		int num_spikes_target = 0;
		
		std::vector<bool> first_spike_source(num_inputs); // indicators that produced spike of the source neuron is the first one
		std::fill(first_spike_source.begin(), first_spike_source.end(), true);
		
		for (int i = 0; i < static_cast<int>(TRIAL_DURATION/TIMESTEP); i++){			
			for (int n = 0; n < num_inputs; n++){
				if (i == ind_kick_times[n])
					source_neurons[n].raiseE(TRAINING_KICK);
					
				source_neurons[n].Debraband_step_no_target_update();
				
				if (source_neurons[n].get_fired_soma()){
					target_neuron.raiseE(input_weights[n]);
					
					if (first_spike_source[n]){
						input_times[n] = static_cast<double>(i) * TIMESTEP;
						first_spike_source[n] = false;
					}
				}
			}
			
			target_neuron.Debraband_step_no_target_update();

			if (target_neuron.get_fired_soma())
			{
				double spike_time = static_cast<double>(i) * TIMESTEP;
				std::cout << "Spike at " << spike_time << std::endl;
				
				if (num_spikes_target == 0) {
					burst_onset_times.push_back(spike_time);
				}
				num_spikes_target += 1;
			}
		}
		
		num_spikes_in_trial.push_back(num_spikes_target);
		burst_onset_times.back() -= std::accumulate(input_times.begin(), input_times.end(), 0.0) / static_cast<double>(num_inputs);
		
		target_neuron.set_to_rest();
		
		for (int n = 0; n < num_inputs; n++)
			source_neurons[n].set_to_rest();
			
	}
	
	std::cout << "Burst onset times:\n"; 
	for (size_t i = 0; i < burst_onset_times.size(); i++)
		std::cout << burst_onset_times[i] << " ";
	std::cout << std::endl;	
	
	std::cout << "Num spikes in burst:\n"; 
	for (size_t i = 0; i < num_spikes_in_trial.size(); i++)
		std::cout << num_spikes_in_trial[i] << " ";
	std::cout << std::endl;		
	
	if (!filename.empty()) write_results_to_file(burst_onset_times, num_spikes_in_trial, filename);
}


int main(int argc, char** argv){
	
	
	int num_inputs = 170; // number of excitatory inputs
	double gee_max = 0.004; // max strenght of excitatory input in mS/cm^2
	double synch_window = 6.0; // time window for input arrivals in ms
	
	int num_trial = 50; // number of trials to repeat
	
	double cm_d = 1.0; // membrane capacitance of dendritic compartment
	std::string filename = ""; // path to output file
	
	if (argc == 3){
		cm_d = atof(argv[1]);
		filename = argv[2];
		
		std::cout << "cm_d = " << cm_d << "\n"
				  << "filename = " << filename << std::endl;
	}
	
	//double mean_input_time = 75.0; // mean input arrival time in ms
	
	//simulate_input(num_inputs, gee_max, mean_input_time, 
					//synch_window, num_trial);
	
	double mean_kick_time = 50.0; // mean kick time of source neurons in ms
	
	simulate_input_from_neurons(num_inputs, gee_max, mean_kick_time, 
					synch_window, num_trial, cm_d, filename);
}

#include "../../../HH2_buffer.h"
#include "../../../poisson_noise.h"
#include <cmath>

#define N 100

int main(int argc, char** argv)
{
	HH2_buffer neurons[N];
	double d_first_spike_times[N];
	double d_dend_spike_times[N];
	bool b_neurons_spiked[N];
	
	for (int i = 0; i < N; i++)
	{
		d_first_spike_times[i] = -1.0;
		d_dend_spike_times[i] = -1.0;
		b_neurons_spiked[i] = false;
	}
	
	Poisson_noise generator;
	
	unsigned seed = 1991;
	
	generator.set_seed(seed);
	
	double white_noise_mean_soma = 0.0;
	double white_noise_std_soma = 0.1;
	double white_noise_mean_dend = 0.0;
	double white_noise_std_dend = 0.2;
	
	double timestep = 0.02;
	double duration = 100;
	
	double kick_time = 50;
	
	int kick_ind = static_cast<int>(kick_time / timestep);
	
	for (int i = 0; i < N; i++)
	{
		neurons[i].set_noise_generator(&generator);
		
		neurons[i].set_white_noise(white_noise_mean_soma, white_noise_std_soma,
								   white_noise_mean_dend, white_noise_std_dend);
								   
		neurons[i].set_dynamics(timestep);
		
	}
	
	double Ge_kick = 3.0;
	
	// evolve dynamics
	
	for (int i = 0; i < (int) duration / timestep; i++)
	{
		for (int j = 0; j < N; j++)
		{
		
			if (i == kick_ind)
				neurons[j].raiseE(Ge_kick);
		
			neurons[j].Debraband_step_no_target_update();

			if ( ( neurons[j].get_fired_soma() ) && ( !b_neurons_spiked[j] ) ) 
			{
				b_neurons_spiked[j] = true;
				d_first_spike_times[j] = static_cast<double>(i) * timestep;
				
		
			}
			
			if ( neurons[j].get_fired_dend() )
				d_dend_spike_times[j] = static_cast<double>(i) * timestep;
				
		}
	}
	
	// show first soma and dend spike times
	
	int num_first_spikes = 0;
	int num_dend_spikes = 0;
	
	double mean_first_spike_time = 0.0;
	double mean_dend_spike_time = 0.0;
	
	double std_first_spike_time = 0.0;
	double std_dend_spike_time = 0.0;
	
	// find mean spike times
	
	for (int i = 0; i < N; i++)
	{
		if ( d_first_spike_times[i] > 0 )
		{
			std::cout << "First spike of neuron " << i << "at " << d_first_spike_times[i] << std::endl;
			
			num_first_spikes++;
			mean_first_spike_time += d_first_spike_times[i];
		}
			
		if ( d_dend_spike_times[i] > 0 )
		{
			std::cout << "Dend spike of neuron " << i << "at " << d_dend_spike_times[i] << std::endl;
			
			num_dend_spikes++;
			mean_dend_spike_time += d_dend_spike_times[i];	
		}
	}
	
	if ( num_first_spikes > 0 )
		mean_first_spike_time /= static_cast<double>(num_first_spikes);
		
	if ( num_dend_spikes > 0 )
		mean_dend_spike_time /= static_cast<double>(num_dend_spikes);
		
	// find std of spike times
	for (int i = 0; i < N; i++)
	{
		if ( ( num_first_spikes > 1 ) && ( d_first_spike_times[i] > 0 ) )
			std_first_spike_time += (mean_first_spike_time - d_first_spike_times[i]) * (mean_first_spike_time - d_first_spike_times[i]);
		
		if ( ( num_dend_spikes > 1 ) && ( d_dend_spike_times[i] > 0 ) )
			std_dend_spike_time += (mean_dend_spike_time - d_dend_spike_times[i]) * (mean_dend_spike_time - d_dend_spike_times[i]);
	}
	
	if ( num_first_spikes > 0 )
		std_first_spike_time = sqrt(std_first_spike_time / static_cast<double>(num_first_spikes - 1));
	
	if ( num_dend_spikes > 0 )
		std_dend_spike_time = sqrt(std_dend_spike_time / static_cast<double>(num_dend_spikes - 1));
		
	std::cout << "Mean first spike time = " << mean_first_spike_time << "\n"
			  << "Std first spike time = "  << std_first_spike_time  << "\n"
			  << "Mean dend spike time = " << mean_dend_spike_time << "\n"
			  << "Std dend spike time = "  << std_dend_spike_time  << std::endl;
			  
	
	
	return 0;
}

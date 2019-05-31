#include <iostream>
#include "HH2.h"
#include "poisson_noise.h"
#include <functional>
#include <fstream>
#include <vector>
#include <sys/stat.h>

using namespace std::placeholders;


const double cm = 0.5;
const double timestep = 0.0125;
	
double I(double ampl, double start, double t)
{
	double duration = 20;
	
	if ( ( t >= start ) && ( t <= (start + duration) ) )
		return ampl;
	else
		return 0;
}

void write_fI(int N, double ampl_step, std::vector<int>& num_spikes, 
						std::vector<int>& num_bursts, const char *filename)
{
	std::ofstream out;
	out.open(filename, std::ios::binary | std::ios::out);
	
	out.write(reinterpret_cast<char*>(&N), sizeof(int));
	out.write(reinterpret_cast<char*>(&ampl_step), sizeof(double));
	
	for (int i = 0; i < N; i++)
	{
		out.write(reinterpret_cast<char*>(&num_spikes[i]), sizeof(int));
		out.write(reinterpret_cast<char*>(&num_bursts[i]), sizeof(int));
	}
	out.close();
}

void write_conductance_response(int N, double G_step,
						std::vector<double>& burst_onset_times, std::vector<double>& spike_times,
						const char *filename)
{
	std::ofstream out;
	out.open(filename, std::ios::binary | std::ios::out);
	
	out.write(reinterpret_cast<char*>(&N), sizeof(int));
	out.write(reinterpret_cast<char*>(&G_step), sizeof(double));
	
	for (int i = 0; i < N; i++)
	{
		out.write(reinterpret_cast<char*>(&burst_onset_times[i]), sizeof(double));
		out.write(reinterpret_cast<char*>(&spike_times[i]), sizeof(double));
	}
	
	out.close();
}

void calculate_fI(int N, double ampl_step, bool injection_to_soma, std::string dirname)
{
	double sim_duration = 100;
	double start = 50.0;
	
	int num_iter = static_cast<int>(sim_duration / timestep);
	
	std::vector<int> num_spikes(N);
	std::vector<int> num_bursts(N);
	
	
	double ampl = 0.0;
	unsigned seed = 1991;
	
	Poisson_noise noise_generator;
	
	noise_generator.set_seed(seed);
	
	std::string filename;
	
	if (injection_to_soma)
		filename = dirname + "fI_soma.bin";
	else
		filename = dirname + "fI_dend.bin";
	
	for (int i = 0; i < N; i++)
	{
		HH2 *neuron = new HH2;
	
	
		std::function<double (double)> f = std::bind(&I, ampl, start, _1);
	
		neuron->set_dynamics(timestep);
		neuron->set_cm_dend(cm);
		
		std::string filename_neuron;
		
		if (injection_to_soma)
			filename_neuron = dirname + "RA_soma_" + std::to_string(i) + ".bin";
		else
			filename_neuron = dirname + "RA_dend_" + std::to_string(i) + ".bin";
		
		struct stat buf;
	
		if ( stat(filename_neuron.c_str(), &buf) == 0 )
			std::remove(filename_neuron.c_str());
		
		
		neuron->set_recording_full(filename_neuron);
		
		if (injection_to_soma)
			neuron->set_soma_current(f);
		else
			neuron->set_dend_current(f);
			
		neuron->set_noise_generator(&noise_generator);
		neuron->set_white_noise(0.0, 0.0, 0.0, 0.0);
	
		for (int j = 0; j < num_iter; j++)
			neuron->Debraband_step_no_target_update();	

		num_spikes[i] = neuron->get_spike_number_soma();
		num_bursts[i] = neuron->get_spike_number_dend();
		
		delete neuron;
		ampl += ampl_step;
	}
	
	write_fI(N, ampl_step, num_spikes, num_bursts, filename.c_str());
}


void calculate_conductance_response(int N, double G_step, std::string dirname)
{
	double sim_duration = 300;
	double kick_time = 50.0;
	bool excited;
	bool spiked;
	
	int num_iter = static_cast<int>(sim_duration / timestep);
	
	std::vector<double> burst_onset_times(N);
	std::vector<double> spike_times(N);
	
	std::fill(burst_onset_times.begin(), burst_onset_times.end(), -1.0);
	std::fill(spike_times.begin(), spike_times.end(), -1.0);
	
	double G = 0.0;
	unsigned seed = 1991;
	
	Poisson_noise noise_generator;
	
	noise_generator.set_seed(seed);
	
	for (int i = 0; i < N; i++)
	{
		HH2 *neuron = new HH2;
	
		neuron->set_cm_dend(cm);
		
		neuron->set_dynamics(timestep);
		
		
		
		neuron->set_noise_generator(&noise_generator);
		//neuron->set_white_noise(0.0, 0.1, 0.0, 0.2);
		neuron->set_white_noise(0.0, 0.0, 0.0, 0.0);
	
		excited = false;
		spiked = false;
	
		std::string filename = dirname + "RA" + std::to_string(i) + ".bin";
		
		struct stat buf;
	
		if ( stat(filename.c_str(), &buf) == 0 )
			std::remove(filename.c_str());
		
		
		neuron->set_recording_full(filename);
		
	
	
		for (int j = 0; j < num_iter; j++)
		{
			neuron->Debraband_step_no_target_update();	
			
			if ( ( static_cast<double>(j) * timestep > kick_time ) && ( !excited ) )
			{
				neuron->raiseE(G);
				excited = true;
			}
			
			if ( neuron->get_fired_dend() )
				burst_onset_times[i] = static_cast<double>(j) * timestep - kick_time;
				
			if ( ( neuron->get_fired_soma() ) && ( !spiked ) )
			{
				spike_times[i] = static_cast<double>(j) * timestep - kick_time;
				spiked = true;
			}
		}
		
		
		delete neuron;
		G += G_step;
	}
	
	std::string filename_G = dirname + "G_response.bin";
	
	write_conductance_response(N, G_step, burst_onset_times, spike_times, filename_G.c_str());
	
}

int main(int argc, char **argv)
{
	std::string dirname = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/neuronResponse/cm0.5/";
	
	int N = 50;
	double ampl_step = 0.025;
	
	
	calculate_fI(N, ampl_step, true, dirname);
	calculate_fI(N, ampl_step, false, dirname);
	
	double G_step = 0.2;
	calculate_conductance_response(N, G_step, dirname);
}

#include <iostream>
#include "../../../HH2_buffer.h"
#include "../../../poisson_noise.h"
#include <string>
#include <functional>
#include <fstream>
#include <algorithm>

#define start 10.0
#define duration 20.0

using namespace std::placeholders;

double Iext(double amplitude, double t)
{
	if ( (t >= start) && (t <= start + duration) )
		return amplitude;
	else 
		return 0.0;
}

void write_data(double I, int num_spikes_soma, int num_spikes_dend, const char * filename)
{
	std::ofstream out;
	
	out.open(filename, std::ios::out | std::ios::binary);

	if (!out.is_open())
		std::cerr << "Couldn't open the file!" << std::endl;

	out.write(reinterpret_cast<char *>(&I), sizeof(I));
	out.write(reinterpret_cast<char *>(&num_spikes_soma), sizeof(num_spikes_soma));
	out.write(reinterpret_cast<char *>(&num_spikes_dend), sizeof(num_spikes_dend));

	out.close();
}

int main(int argc, char** argv)
{
	double I_amplitude; // amplitude of injected current
	double timestep; // dynamics timestep in ms
	std::string filename; // filename where data output goes
	int soma_injection; // 1 if current is injected to soma, 0 if to dendrite

	if (argc > 1)
	{
		I_amplitude = atof(argv[1]);
		timestep = atof(argv[2]);
		soma_injection = atoi(argv[3]);
		filename = argv[4];

		std::cout << "I_amplitude = " << I_amplitude << std::endl;
		std::cout << "timestep = " << timestep << std::endl;
		std::cout << "soma_injection = " << soma_injection << std::endl;
		std::cout << "filename = " << filename << std::endl;

	}

	else
	{
		std::cerr << "No command line arguments were supplied" << std::endl;
		return -1;
	}
	
	HH2_buffer n; // excitatory neuron

	double interval = 200.0; // simulation interval in ms

	std::function<double (double)> I = std::bind(&Iext, I_amplitude, _1);

	Poisson_noise noise_generator; // noise generator
	unsigned seed = 1991; // seed for noise generator

	noise_generator.set_seed(seed);

	n.set_noise_generator(&noise_generator); // set noise generator
	n.set_dynamics(timestep); // set neuron dynamics
	
	if (soma_injection)
		n.set_soma_current(I); // set injected current
	else
		n.set_dend_current(I);

	int num_steps = static_cast<int>(round(interval / timestep)); // number of steps to perform

	// evolve dynamics
	for (int i = 0; i < num_steps; i++)
		n.Debraband_step_no_target_update();
		//n.R4_step_no_target_update();

	int num_spikes_soma = n.get_spike_number_soma(); // number of somatic spikes
	int num_spikes_dend = n.get_spike_number_dend(); // number of dendritic spikes

	write_data(I_amplitude, num_spikes_soma, num_spikes_dend, filename.c_str()); // write dynamics to file

	return 0;
}

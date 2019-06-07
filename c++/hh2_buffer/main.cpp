#include "HH2.h"
#include <string>
#include <functional>
#include "poisson_noise.h"

#define START 50.0
#define DURATION 20.0
#define AMPLITUDE 0.0

double I(double t)
{
	if ( (t >= START) && (t <= START + DURATION) )
		return AMPLITUDE;
	else
		return 0.0;
}


int main()
{
	HH2 n;
	Poisson_noise noise_generator;
	
	unsigned seed = 1991;
	
	noise_generator.set_seed(seed);  

	double duration = 100000.0;

	//double timestep = 0.001;
	double timestep = 0.02;

	std::string filename = "/home/eugene/Programming/data/mlong/noise/noiseCheckDebrabandNew/noise_s0.30_d0.0_dt0.02.bin";
	//std::string filename = "/home/eugene/Programming/data/mlong/noise/noiseCheckEuler/noise_s0.1_d0.2_dt0.01.bin";
	
	//std::function<double(double)> Iext
	// set noise
	double white_noise_mean_soma = 0.0;
	double white_noise_std_soma = 0.30;
	double white_noise_mean_dend = 0.0;
	double white_noise_std_dend = 0.0;
	
	n.set_noise_generator(&noise_generator);
	//n.set_poisson_noise();
	n.set_white_noise(white_noise_mean_soma, white_noise_std_soma,
					  white_noise_mean_dend, white_noise_std_dend);
	
	//n.set_dend_current(&I);
	n.set_dynamics(timestep);
	n.set_recording(filename);
	
	double Ge_kick = 0.0;
	
	
	for (int i = 0; i < (int) duration / timestep; i++)
	{
		if (i == 5000)
			n.raiseE(Ge_kick);
		//n.R4_step_no_target_update();
		n.Debraband_step_no_target_update();
		//n.Euler_Maruyama_step_no_target_update();

		if (n.get_fired_soma())
			std::cout << "Spike at " << static_cast<double>(i) * timestep << std::endl;

		if (n.get_fired_dend())
			std::cout << "Burst at " << static_cast<double>(i) * timestep << std::endl;

	}
}

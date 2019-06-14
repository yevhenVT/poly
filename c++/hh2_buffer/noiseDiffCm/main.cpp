#include "HH2.h"
#include <string>
#include "poisson_noise.h"

int main(int argc, char** argv)
{
	HH2 n;
	Poisson_noise noise_generator;
	
	unsigned seed = 1991;
	
	noise_generator.set_seed(seed);  

	double duration = 100000.0;

	//double timestep = 0.001;
	double timestep = 0.01;
	
	double cm_d = 1.0; // membrane capacitance of dendritic compartment
	std::string filename = ""; // path to output file
	
	double white_noise_mean_soma;
	double white_noise_std_soma;
	double white_noise_mean_dend;
	double white_noise_std_dend;
	
	
	if (argc == 7){
		cm_d = atof(argv[1]);
		white_noise_mean_soma = atof(argv[2]);
		white_noise_std_soma = atof(argv[3]);
		white_noise_mean_dend = atof(argv[4]);
		white_noise_std_dend = atof(argv[5]);
		
		filename = argv[6];
		
		std::cout << "cm_d = " << cm_d << "\n"
				  << "white_noise_mean_soma = " << white_noise_mean_soma << "\n"
				  << "white_noise_std_soma = "  << white_noise_std_soma << "\n"
				  << "white_noise_mean_dend = " << white_noise_mean_dend << "\n"
				  << "white_noise_std_dend = "  << white_noise_std_dend << "\n"
				  << "filename = " << filename  << std::endl;
	}
	else{
		std::cout << "Not enough parameters!" << std::endl;
		return 0;
	}
	
	// set noise
	
	n.set_noise_generator(&noise_generator);
	//n.set_poisson_noise();
	n.set_white_noise(white_noise_mean_soma, white_noise_std_soma,
					  white_noise_mean_dend, white_noise_std_dend);
	
	n.set_cm_dend(cm_d);
	n.set_dynamics(timestep);
	n.set_recording(filename);
	
	
	for (int i = 0; i < (int) duration / timestep; i++)
	{
		//n.R4_step_no_target_update();
		n.Debraband_step_no_target_update();
		//n.Euler_Maruyama_step_no_target_update();

		if (n.get_fired_soma())
			std::cout << "Spike at " << static_cast<double>(i) * timestep << std::endl;

		if (n.get_fired_dend())
			std::cout << "Burst at " << static_cast<double>(i) * timestep << std::endl;
	}
}

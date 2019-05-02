#pragma once
#ifndef HHI_H
#define HHI_H

#include <vector>
#include <functional>
#include <cmath>

class Poisson_noise;

using std::vector;
typedef std::function<double (double)> DDfunction;

class HHI
{
public:
	HHI();
	
	// print
	void print_param(); // print parameters of the model
	
	// get internal states of dynamics
	int get_spike_number();	//	get number of action potential spikes
	bool get_fired(); // get if neuron fired AP
	
	// calculate dynamics
	void R4_step_no_target_update(); // do one step of RK4 without update of target conductances
	void Runge4_step();	//	do one step of Runge-Kutta order 4

	void DP8_step_no_target_update(); // do one step of Dormand Prince order 8 without update of target conductances
	void DP8_step(); // do one step of Dormand Prince order 8 method 
	
	// noise
	void set_no_poisson_noise(); //	turn off Poisson noise
	void set_no_white_noise(); // turn off white noise
    void set_noise_generator(Poisson_noise* g);	//	set pointer to Poisson noise generator 
	void set_white_noise(double m, double s); // set white-noise injected current
    void set_poisson_noise(); // enable Poisson noise

	// miscalleneous
	void set_recording(std::string filename); // set filename for the data to be saved
	void set_dynamics(double tS);	//	function to set the dynamic range of neuron (time step and time interval)
	
	void raiseE(double G);	//	increase excitatory conductance of the neuron due to excitatory spike
	void raiseI(double G);	//	increase inhibitory conductance of the neuron due to inhibitory spike

	void set_injected_current(DDfunction f); // set a function for injected current
	
	void set_to_rest(); // set all variables to the resting state
	void reset_time(); // reset neuron activity. Last values for all variables are assinged to the first elements 
	
	
	
protected:
	// constants and parameters of the cell
	
	const static double cm;	//	membrane capacitance
	const static double A;	//	neuron's area

	const static double Ena;	//	equilibrium potential of sodium ions
	const static double Ek;	//	equilibrium potential of potassium ions
	const static double El;	//	equilibrim membrane potential
	const static double Ei;	//	inhibitory reversal potential

	const static double gNa;	//	maximum sodium channels conductance (all open)
	const static double gKdr;	//	delay-rectified potassium channel conductance 
	const static double gKHT;	//	high-treshold potassium channel conductance
	const static double gL;	//	membrane leakage conductance
	
	const static double tExc;	//	time scale of excitatory conductance decay
	const static double tInh;	//	time scale of inhibitory conductance decay

	const static double threshold;	//	threshold indicator for action potential generator
	const static double spike_margin; // margin in mV which says when the spike is over (voltage < threshold - spike_margin)
	
	// buffer
	const static int BUFFER_SIZE; // size of the buffer to store data for dumping to file
	int _buffer_counter; // internal counter used to check buffer overfill
	
	// dynamics variables
	
	double _voltage; // neuron voltage
	double _time; // time
	
	double _n;		// vector that contains values of gating variable of delay-rectified potassium (K+) channel
	double _m;		// vector that contains values of activation gating variable of sodium (Na+) channel
	double _h;		// vector that contains values of inactivation gating variable of sodium (Na+) channel
	double _w;		// vector that contains values of gating variable of high-threshold potassium (K+) channel 
	double _I;		// vector that contains injected current values
	double _Gexc;	//	vector that contains total conductance of excitatory synapses
	double _Ginh;	//	vector that contains total conductance of inhibitory synapses
	
	// buffers
	std::string _filename; // name of the file for dumping data
	 
	std::vector<double> buffer_time;
	std::vector<double> buffer_V;
	std::vector<double> buffer_Ge;
	std::vector<double> buffer_Gi;
	
	// supportive internal state parameters of dynamics
	bool _recorded; // indicator that neuron state is recorded
	
	int _flag;	//	vector-indicator of crossing threshold
	int _Nspikes;	//	number of spikes occured during dynamics
	bool _fired; // state of neuron (fired/silent)
	double _timestep;	//	time step for solving the dynamics of the model
	
	// internal state check functions
	
	void noise_check(double& G, double& noise_time); // check noise
	void state_check(); // check if neuron fired
	void state_noise_check(); // check both states
	
	//	conductance
	double Gi(double t);	//	calculate inhibitory conductance at time point t
	double Ge(double t);	//	calculate excitatory conductance at time point t
	
	//	Noise
	Poisson_noise* _generator;	//	pointer to Poisson noise generator

    // poisson noise
	bool _poisson_noise;	//	indicator for Poisson noise
	double _noise_inh;	//	time of inhibitory noisy spike rounded to the time grid
	double _noise_exc;	//	time of excitatory noisy spike rounded to the time grid
	const static double G_noise;	//	maximum noise conductance added to either inhibitory							//	or excitatory conductance of neuron
	const static double lambda;	//	parameter of Poisson point process
	
	void initialize_noise(double& noise_time); // initialize noise spike times

	//	current
	bool _injected_current; // indicator that external current is injected 
	double I_total(double t); // total current 
	DDfunction _Iinjected; // injected current in nA

	double I_default(double t){return 0;}; // default function for injected current (returns zero)	
	
	// write and record
	void write_buffers(); // flush buffer to a file

	void record_state(); // record neuron state variables
	
	// support functions for Runge-Kutta method
	double kV(double v, double t, double h, double w, double m3, double n4);
	double kn(double v, double n);
	double km(double v, double m);
	double kh(double v, double h);
	double kw(double v, double w);

	static double an(double v){return 0.15*(v + 15) / (1 - exp(-(v + 15) / 10));} // was 0.05; original value = 0.15
	static double bn(double v){return 0.2 * exp(-(v + 25) / 80);} // was 0.1; original value = 0.2
	static double am(double v){return (v + 22) / (1 - exp(-(v + 22) / 10));}
	static double bm(double v){return 40 * exp(-(v + 47) / 18);}
	static double ah(double v){return 0.7 * exp(-(v + 34) / 20);}
	static double bh(double v){return 10 / (1 + exp(-(v + 4) / 10));}
	static double wInf(double v){return 1 / (1 + exp(-v / 5));}
	static double tauW(double v){return 1;} // was 2; original value = 1
};



#endif

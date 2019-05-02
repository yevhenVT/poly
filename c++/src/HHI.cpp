#include "HHI.h"
#include "poisson_noise.h"
#include <fstream>
#include <iostream>
#include "exception.h"
#include <algorithm>
#include <cmath>

using namespace std::placeholders;

// neuron model parameters
const double HHI::cm = 1.0;	// micro F / cm2
const double HHI::A = 6000;	// microns2

const double HHI::Ena = 55.0;	// mV
const double HHI::Ek = -80.0;
const double HHI::El = -65.0;
const double HHI::Ei = -75.0;

const double HHI::gNa = 100.0;	// mS/cm2
const double HHI::gKdr = 20.0;
const double HHI::gKHT = 500.0;
const double HHI::gL = 0.1;

const double HHI::tExc = 2.0;	//	ms
const double HHI::tInh = 5.0;

const double HHI::threshold = -20.0;	//	mV

const double HHI::spike_margin = 5.0; // mV

const int HHI::BUFFER_SIZE = 1000; // size of buffer for data

// noise parameters
const double HHI::G_noise = 0.45;	//	maximum noise conductance
const double HHI::lambda = 250.0; // intensity parameter for Poisson noise

HHI::HHI()
{
	// internal state
	_injected_current = false;
	
	_noise_exc = 0.0;
	_noise_inh = 0.0;
	_timestep = 0.0;

	// noise
	_generator = nullptr;
	
	_recorded = false;
	_fired = false;
	
	_buffer_counter = 0;

    // Poisson noise
	_poisson_noise = false;	//	turn on the noise

	this->set_to_rest(); // initialize internal state of neuron
}

void HHI::print_param()
{
	std::cout << "Cm = " << cm << " microF / cm^2" << std::endl;

	std::cout << "EL = " << El << " mV" << std::endl;
	std::cout << "EK = " << Ek << " mV" << std::endl;
	std::cout << "ENa = " << Ena << " mV" << std::endl;
	std::cout << "Ei = " << Ei << " mV" << std::endl;

	std::cout << "gL = " << gL << " mS / cm^2" << std::endl;
	std::cout << "gKdr = " << gKdr << " mS / cm^2" << std::endl;
	std::cout << "gKHT = " << gKHT << " mS / cm^2" << std::endl;
	std::cout << "gNa = " << gNa << " mS / cm^2" << std::endl;

	std::cout << "tExc = " << tExc << " ms" << std::endl;
	std::cout << "tInh = " << tInh << " ms" << std::endl;
}

bool HHI::get_fired()
{
	return _fired;
}

int HHI::get_spike_number()
{
	return _Nspikes;
}

void HHI::set_injected_current(DDfunction f)
{
	_Iinjected = f;
	_injected_current = true;
}

void HHI::set_noise_generator(Poisson_noise* g)
{
	_generator = g;
}

void HHI::set_poisson_noise()
{
    _poisson_noise = true;
            
    this->initialize_noise(_noise_exc);
	this->initialize_noise(_noise_inh);
}


void HHI::set_no_poisson_noise()
{
	_poisson_noise = false;
}

void HHI::set_recording(std::string filename)
{
	_recorded = true;
	_filename = filename;
	
	buffer_time.resize(BUFFER_SIZE);
	buffer_V.resize(BUFFER_SIZE);
	buffer_Ge.resize(BUFFER_SIZE);
	buffer_Gi.resize(BUFFER_SIZE);
}

void HHI::record_state()
{
	buffer_time[_buffer_counter] = _time;
	buffer_V[_buffer_counter] = _voltage;
	buffer_Ge[_buffer_counter] = _Gexc;
	buffer_Gi[_buffer_counter] = _Ginh;
	
	//std::cout << "_time = " << _time << "\n";
	//std::cout << "itime = " << itime << "\n";

	//std::cout << "buffer_time[itime] = " << buffer_time[itime] << "\n" << std::endl;

	_buffer_counter += 1;
	
	// if buffer is full, write it to the file
	if (_buffer_counter == BUFFER_SIZE)
	{
		this->write_buffers();
		_buffer_counter = 0;
	}
}

void HHI::set_dynamics(double tS)
{

	_timestep = tS;
	

}

void HHI::set_to_rest()
{
	_time = 0.0;
	_voltage = -66;
	_n = 0.125;
	_m = 0.0;
	_h = 0.99;
	_w = 0.0;
	_Gexc = 0.0;
	_Ginh = 0.0;
	_flag = 0;
	
	_Nspikes = 0;
	
	_I = I_total(_time);

	//	set up noise
    if (_poisson_noise)
    {
	    this->initialize_noise(_noise_exc);
	    this->initialize_noise(_noise_inh);
    }
}

void HHI::reset_time()
{

	_time = 0.0;
	
	_I = I_total(_time);

	//	set up noise
    if (_poisson_noise)
    {
		this->initialize_noise(_noise_exc);
		this->initialize_noise(_noise_inh);
    }
}

void HHI::initialize_noise(double& noise_time)
{
    try
    {
        if (_generator == nullptr)
        {
            throw NoGenerator("Noise generator is not set for HHI neuron!\n");
        }
        else
        {
        	noise_time = 1000 * _generator->get_spike_time(lambda);
	        
            while (noise_time < _timestep)
		        noise_time = 1000 * _generator->get_spike_time(lambda);
        }
    }

    catch (NoGenerator const& e)
    {
        std::cerr << "NoGenerator Exception: " << e.what() << std::endl;
    }

}

void HHI::state_noise_check()
{
	this->state_check();

	if (_poisson_noise)
	{
		this->noise_check(_Gexc, _noise_exc);
		this->noise_check(_Ginh, _noise_inh);
	}

}

void HHI::state_check()
{
	if ( ( _flag == 1 ) && ( _voltage < threshold - spike_margin) )
	{
		_flag = 0;
		_Nspikes++;
		_fired = true;
	}
	else
	{
		_fired = false;
		//	check if we should change the state of neuron (voltage crossed the threshold)
		if ( ( _flag == 0) && ( _voltage > threshold ) )
			_flag = 1;
	}

}

void HHI::noise_check(double& G, double& noise_time)
{
	if (_time > noise_time)
	{
		G += _generator->random(G_noise);

		double random = 1000 * _generator->get_spike_time(lambda);
		
		while (random < _timestep)
		{
				random = 1000 * _generator->get_spike_time(lambda);
				G += _generator->random(G_noise);
		}
		noise_time = noise_time + random;
	}

}

void HHI::R4_step_no_target_update()
{
	if (_recorded)
		this->record_state();
		
	this->state_noise_check();
	this->Runge4_step();
}

void HHI::DP8_step_no_target_update()
{
	if (_recorded)
		this->record_state();
		
	this->state_noise_check();
	this->DP8_step();
}

void HHI::Runge4_step()
{
	double m1, n1, h1, w1, m3, n4;
	double v, t;
	double k1V, k2V, k3V, k4V;
	double k1n, k2n, k3n, k4n;
	double k1m, k2m, k3m, k4m;
	double k1h, k2h, k3h, k4h;
	double k1w, k2w, k3w, k4w;

	m1 = _m;
	n1 = _n;
	h1 = _h;
	w1 = _w;
	v  = _voltage;
	t  = _time;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k1V = kV(v, t, h1, w1, m3, n4);
	k1n = kn(v, n1);
	k1m = km(v, m1);
	k1h = kh(v, h1);
	k1w = kw(v, w1);

	m1 = _m + _timestep * k1m / 3;
	n1 = _n + _timestep * k1n / 3;
	h1 = _h + _timestep * k1h / 3;
	w1 = _w + _timestep * k1w / 3;
	v  = _voltage + _timestep * k1V / 3;
	t  = _time + _timestep / 3;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k2V = kV(v, t, h1, w1, m3, n4);
	k2n = kn(v, n1);
	k2m = km(v, m1);
	k2h = kh(v, h1);
	k2w = kw(v, w1);

	m1 = _m + _timestep * (-k1m / 3 + k2m);
	n1 = _n + _timestep * (-k1n / 3 + k2n);
	h1 = _h + _timestep * (-k1h / 3 + k2h);
	w1 = _w + _timestep * (-k1w / 3 + k2w);
	v  = _voltage + _timestep * (-k1V / 3 + k2V);
	t  = _time + 2 * _timestep / 3;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k3V = kV(v, t, h1, w1, m3, n4);
	k3n = kn(v, n1);
	k3m = km(v, m1);
	k3h = kh(v, h1);
	k3w = kw(v, w1);

	m1 = _m + _timestep * (k1m - k2m + k3m);
	n1 = _n + _timestep * (k1n - k2n + k3n);
	h1 = _h + _timestep * (k1h - k2h + k3h);
	w1 = _w + _timestep * (k1w - k2w + k3w);
	v  = _voltage + _timestep * (k1V - k2V + k3V);
	t  = _time + _timestep;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k4V = kV(v, t, h1, w1, m3, n4);
	k4n = kn(v, n1);
	k4m = km(v, m1);
	k4h = kh(v, h1);
	k4w = kw(v, w1);

	//	update all values for next time point

	_voltage = _voltage + _timestep * (k1V + 3 * k2V + 3 * k3V + k4V) / 8;
	_n = _n + _timestep * (k1n + 3 * k2n + 3 * k3n + k4n) / 8;
	_m = _m + _timestep * (k1m + 3 * k2m + 3 * k3m + k4m) / 8;
	_h = _h + _timestep * (k1h + 3 * k2h + 3 * k3h + k4h) / 8;
	_w = _w + _timestep * (k1w + 3 * k2w + 3 * k3w + k4w) / 8;
	_Gexc = Ge(_time + _timestep);
	_Ginh = Gi(_time + _timestep);
	
	_time = _time + _timestep;
	
	_I = I_total(_time);
}

void HHI::DP8_step()
{

	double m1, n1, h1, w1, m3, n4;
	double v, t;
	double k1V, k2V, k3V, k4V, k5V, k6V, k7V, k8V, k9V, k10V, k11V, k12V, k13V;
	double k1n, k2n, k3n, k4n, k5n, k6n, k7n, k8n, k9n, k10n, k11n, k12n, k13n;
	double k1m, k2m, k3m, k4m, k5m, k6m, k7m, k8m, k9m, k10m, k11m, k12m, k13m;
	double k1h, k2h, k3h, k4h, k5h, k6h, k7h, k8h, k9h, k10h, k11h, k12h, k13h;
	double k1w, k2w, k3w, k4w, k5w, k6w, k7w, k8w, k9w, k10w, k11w, k12w, k13w;

	m1 = _m;
	n1 = _n;
	h1 = _h;
	w1 = _w;
	v = _voltage;
	t = _time;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k1V = kV(v, t, h1, w1, m3, n4);
	k1n = kn(v, n1);
	k1m = km(v, m1);
	k1h = kh(v, h1);
	k1w = kw(v, w1);

	m1 = _m + _timestep * k1m / 18;
	n1 = _n + _timestep * k1n / 18;
	h1 = _h + _timestep * k1h / 18;
	w1 = _w + _timestep * k1w / 18;
	v = _voltage + _timestep * k1V / 18;
	t = _time + _timestep / 18;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k2V = kV(v, t, h1, w1, m3, n4);
	k2n = kn(v, n1);
	k2m = km(v, m1);
	k2h = kh(v, h1);
	k2w = kw(v, w1);

	m1 = _m + _timestep * (k1m + 3 * k2m) / 48;
	n1 = _n + _timestep * (k1n + 3 * k2n) / 48;
	h1 = _h + _timestep * (k1h + 3 * k2h) / 48;
	w1 = _w + _timestep * (k1w + 3 * k2w) / 48;
	v = _voltage + _timestep * (k1V + 3 * k2V) / 48;
	t = _time + _timestep / 12;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k3V = kV(v, t, h1, w1, m3, n4);
	k3n = kn(v, n1);
	k3m = km(v, m1);
	k3h = kh(v, h1);
	k3w = kw(v, w1);

	m1 = _m + _timestep * (k1m + 3 * k3m) / 32;
	n1 = _n + _timestep * (k1n + 3 * k3n) / 32;
	h1 = _h + _timestep * (k1h + 3 * k3h) / 32;
	w1 = _w + _timestep * (k1w + 3 * k3w) / 32;
	v = _voltage + _timestep * (k1V + 3 * k3V) / 32;
	t = _time + _timestep / 8;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k4V = kV(v, t, h1, w1, m3, n4);
	k4n = kn(v, n1);
	k4m = km(v, m1);
	k4h = kh(v, h1);
	k4w = kw(v, w1);

	m1 = _m + _timestep * (20 * k1m - 75 * k3m + 75 * k4m) / 64;
	n1 = _n + _timestep * (20 * k1n - 75 * k3n + 75 * k4n) / 64;
	h1 = _h + _timestep * (20 * k1h - 75 * k3h + 75 * k4h) / 64;
	w1 = _w + _timestep * (20 * k1w - 75 * k3w + 75 * k4w) / 64;
	v = _voltage + _timestep * (20 * k1V - 75 * k3V + 75 * k4V) / 64;
	t = _time + 5 * _timestep / 16;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k5V = kV(v, t, h1, w1, m3, n4);
	k5n = kn(v, n1);
	k5m = km(v, m1);
	k5h = kh(v, h1);
	k5w = kw(v, w1);
	
	m1 = _m + _timestep * (3 * k1m + 15 * k4m + 12 * k5m) / 80;
	n1 = _n + _timestep * (3 * k1n + 15 * k4n + 12 * k5n) / 80;
	h1 = _h + _timestep * (3 * k1h + 15 * k4h + 12 * k5h) / 80;
	w1 = _w + _timestep * (3 * k1w + 15 * k4w + 12 * k5w) / 80;
	v = _voltage + _timestep * (3 * k1V + 15 * k4V + 12 * k5V) / 80;
	t = _time + 3 * _timestep / 8;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k6V = kV(v, t, h1, w1, m3, n4);
	k6n = kn(v, n1);
	k6m = km(v, m1);
	k6h = kh(v, h1);
	k6w = kw(v, w1);
	
	m1 = _m + _timestep * (29443841 * k1m / 614563906 + 77736538 * k4m / 692538347 - 28693883 * k5m / 1125000000 + 23124283 * k6m / 1800000000);
	n1 = _n + _timestep * (29443841 * k1n / 614563906 + 77736538 * k4n / 692538347 - 28693883 * k5n / 1125000000 + 23124283 * k6n / 1800000000);
	h1 = _h + _timestep * (29443841 * k1h / 614563906 + 77736538 * k4h / 692538347 - 28693883 * k5h / 1125000000 + 23124283 * k6h / 1800000000);
	w1 = _w + _timestep * (29443841 * k1w / 614563906 + 77736538 * k4w / 692538347 - 28693883 * k5w / 1125000000 + 23124283 * k6w / 1800000000);
	v = _voltage + _timestep * (29443841 * k1V / 614563906 + 77736538 * k4V / 692538347 - 28693883 * k5V / 1125000000 + 23124283 * k6V / 1800000000);
	t = _time + 59 * _timestep / 400;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k7V = kV(v, t, h1, w1, m3, n4);
	k7n = kn(v, n1);
	k7m = km(v, m1);
	k7h = kh(v, h1);
	k7w = kw(v, w1);
	
	m1 = _m + _timestep * (16016141 * k1m / 946692911 + 61564180 * k4m / 158732637 + 22789713 * k5m / 633445777 + 545815736 * k6m / 2771057229 - 180193667 * k7m / 1043307555);
	
	n1 = _n + _timestep * (16016141 * k1n / 946692911 + 61564180 * k4n / 158732637 + 22789713 * k5n / 633445777 + 545815736 * k6n / 2771057229 - 180193667 * k7n / 1043307555);
	
	h1 = _h + _timestep * (16016141 * k1h / 946692911 + 61564180 * k4h / 158732637 + 22789713 * k5h / 633445777 + 545815736 * k6h / 2771057229 - 180193667 * k7h / 1043307555);
	
	w1 = _w + _timestep * (16016141 * k1w / 946692911 + 61564180 * k4w / 158732637 + 22789713 * k5w / 633445777 + 545815736 * k6w / 2771057229 - 180193667 * k7w / 1043307555);
	
	v  = _voltage + _timestep * (16016141 * k1V / 946692911 + 61564180 * k4V / 158732637 + 22789713 * k5V / 633445777 + 545815736 * k6V / 2771057229 - 180193667 * k7V / 1043307555);
	
	t = _time + 93 * _timestep / 200;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k8V = kV(v, t, h1, w1, m3, n4);
	k8n = kn(v, n1);
	k8m = km(v, m1);
	k8h = kh(v, h1);
	k8w = kw(v, w1);
	
	m1 = _m + _timestep * (39632708 * k1m / 573591083 - 433636366 * k4m / 683701615 - 421739975 * k5m / 2616292301 + 100302831 * k6m / 723423059 + 790204164 * k7m / 839813087 + 800635310 * k8m / 3783071287);
	
	n1 = _n + _timestep * (39632708 * k1n / 573591083 - 433636366 * k4n / 683701615 - 421739975 * k5n / 2616292301 + 100302831 * k6n / 723423059 + 790204164 * k7n / 839813087 + 800635310 * k8n / 3783071287);
	
	h1 = _h + _timestep * (39632708 * k1h / 573591083 - 433636366 * k4h / 683701615 - 421739975 * k5h / 2616292301 + 100302831 * k6h / 723423059 + 790204164 * k7h / 839813087 + 800635310 * k8h / 3783071287);
	
	w1 = _w + _timestep * (39632708 * k1w / 573591083 - 433636366 * k4w / 683701615 - 421739975 * k5w / 2616292301 + 100302831 * k6w / 723423059 + 790204164 * k7w / 839813087 + 800635310 * k8w / 3783071287);
	
	v  = _voltage + _timestep * (39632708 * k1V / 573591083 - 433636366 * k4V / 683701615 - 421739975 * k5V / 2616292301 + 100302831 * k6V / 723423059 + 790204164 * k7V / 839813087 + 800635310 * k8V / 3783071287);
	
	t = _time + 5490023248 * _timestep / 9719169821;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k9V = kV(v, t, h1, w1, m3, n4);
	k9n = kn(v, n1);
	k9m = km(v, m1);
	k9h = kh(v, h1);
	k9w = kw(v, w1);
	
	m1 = _m + _timestep * (246121993 * k1m / 1340847787 - 37695042795 * k4m / 15268766246 - 309121744 * k5m / 1061227803 - 12992083 * k6m / 490766935 + 6005943493 * k7m / 2108947869 + 393006217 * k8m / 1396673457 + 123872331 * k9m / 1001029789);
	
	n1 = _n + _timestep * (246121993 * k1n / 1340847787 - 37695042795 * k4n / 15268766246 - 309121744 * k5n / 1061227803 - 12992083 * k6n / 490766935 + 6005943493 * k7n / 2108947869 + 393006217 * k8n / 1396673457 + 123872331 * k9n / 1001029789);
	
	h1 = _h + _timestep * (246121993 * k1h / 1340847787 - 37695042795 * k4h / 15268766246 - 309121744 * k5h / 1061227803 - 12992083 * k6h / 490766935 + 6005943493 * k7h / 2108947869 + 393006217 * k8h / 1396673457 + 123872331 * k9h / 1001029789);
	
	w1 = _w + _timestep * (246121993 * k1w / 1340847787 - 37695042795 * k4w / 15268766246 - 309121744 * k5w / 1061227803 - 12992083 * k6w / 490766935 + 6005943493 * k7w / 2108947869 + 393006217 * k8w / 1396673457 + 123872331 * k9w / 1001029789);
	
	v  = _voltage + _timestep * (246121993 * k1V / 1340847787 - 37695042795 * k4V / 15268766246 - 309121744 * k5V / 1061227803 - 12992083 * k6V / 490766935 + 6005943493 * k7V / 2108947869 + 393006217 * k8V / 1396673457 + 123872331 * k9V / 1001029789);
	
	t = _time + 13 * _timestep / 20;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k10V = kV(v, t, h1, w1, m3, n4);
	k10n = kn(v, n1);
	k10m = km(v, m1);
	k10h = kh(v, h1);
	k10w = kw(v, w1);
	
	m1 = _m + _timestep * (- 1028468189 * k1m / 846180014 + 8478235783 * k4m / 508512852 + 1311729495 * k5m / 1432422823 - 10304129995 * k6m / 1701304382 - 48777925059 * k7m / 3047939560 + 15336726248 * k8m / 1032824649 - 45442868181 * k9m / 3398467696 + 3065993473 * k10m / 597172653);
	
	n1 = _n + _timestep * (- 1028468189 * k1n / 846180014 + 8478235783 * k4n / 508512852 + 1311729495 * k5n / 1432422823 - 10304129995 * k6n / 1701304382 - 48777925059 * k7n / 3047939560 + 15336726248 * k8n / 1032824649 - 45442868181 * k9n / 3398467696 + 3065993473 * k10n / 597172653);
	
	h1 = _h + _timestep * (- 1028468189 * k1h / 846180014 + 8478235783 * k4h / 508512852 + 1311729495 * k5h / 1432422823 - 10304129995 * k6h / 1701304382 - 48777925059 * k7h / 3047939560 + 15336726248 * k8h / 1032824649 - 45442868181 * k9h / 3398467696 + 3065993473 * k10h / 597172653);
	
	w1 = _w + _timestep * (- 1028468189 * k1w / 846180014 + 8478235783 * k4w / 508512852 + 1311729495 * k5w / 1432422823 - 10304129995 * k6w / 1701304382 - 48777925059 * k7w / 3047939560 + 15336726248 * k8w / 1032824649 - 45442868181 * k9w / 3398467696 + 3065993473 * k10w / 597172653);
	
	v = _voltage + _timestep * (- 1028468189 * k1V / 846180014 + 8478235783 * k4V / 508512852 + 1311729495 * k5V / 1432422823 - 10304129995 * k6V / 1701304382 - 48777925059 * k7V / 3047939560 + 15336726248 * k8V / 1032824649 - 45442868181 * k9V / 3398467696 + 3065993473 * k10V / 597172653);
	
	t = _time + 1201146811 * _timestep / 12990119798;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k11V = kV(v, t, h1, w1, m3, n4);
	k11n = kn(v, n1);
	k11m = km(v, m1);
	k11h = kh(v, h1);
	k11w = kw(v, w1);
	
	m1 = _m + _timestep * (185892177 * k1m / 718116043 - 3185094517 * k4m / 667107341 - 477755414 * k5m / 1098053517 - 703635378 * k6m / 230739211 + 5731566787 * k7m / 1027545527 + 5232866602 * k8m / 850066563 - 4093664535 * k9m / 808688257 + 3962137247 * k10m / 1805957418 + 65686358 * k11m / 487910083);
	
	n1 = _n + _timestep * (185892177 * k1n / 718116043 - 3185094517 * k4n / 667107341 - 477755414 * k5n / 1098053517 - 703635378 * k6n / 230739211 + 5731566787 * k7n / 1027545527 + 5232866602 * k8n / 850066563 - 4093664535 * k9n / 808688257 + 3962137247 * k10n / 1805957418 + 65686358 * k11n / 487910083);
	
	h1 = _h + _timestep * (185892177 * k1h / 718116043 - 3185094517 * k4h / 667107341 - 477755414 * k5h / 1098053517 - 703635378 * k6h / 230739211 + 5731566787 * k7h / 1027545527 + 5232866602 * k8h / 850066563 - 4093664535 * k9h / 808688257 + 3962137247 * k10h / 1805957418 + 65686358 * k11h / 487910083);
	
	w1 = _w + _timestep * (185892177 * k1w / 718116043 - 3185094517 * k4w / 667107341 - 477755414 * k5w / 1098053517 - 703635378 * k6w / 230739211 + 5731566787 * k7w / 1027545527 + 5232866602 * k8w / 850066563 - 4093664535 * k9w / 808688257 + 3962137247 * k10w / 1805957418 + 65686358 * k11w / 487910083);
	
	v  = _voltage + _timestep * (185892177 * k1V / 718116043 - 3185094517 * k4V / 667107341 - 477755414 * k5V / 1098053517 - 703635378 * k6V / 230739211 + 5731566787 * k7V / 1027545527 + 5232866602 * k8V / 850066563 - 4093664535 * k9V / 808688257 + 3962137247 * k10V / 1805957418 + 65686358 * k11V / 487910083);
	
	t = _time + _timestep;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k12V = kV(v, t, h1, w1, m3, n4);
	k12n = kn(v, n1);
	k12m = km(v, m1);
	k12h = kh(v, h1);
	k12w = kw(v, w1);

	/*
	m1 = _m + _timestep * (403863854 * k1m / 491063109 - 5068492393 * k4m / 434740067 - 411421997 * k5m / 543043805 + 652783627 * k6m / 914296604 + 11173962825 * k7m / 925320556 - 13158990841 * k8m / 6184727034 + 3936647629 * k9m / 1978049680 - 160528059 * k10m / 685178525 + 248638103 * k11m / 1413531060;
	
	n1 = _n + _timestep * (403863854 * k1n / 491063109 - 5068492393 * k4n / 434740067 - 411421997 * k5n / 543043805 + 652783627 * k6n / 914296604 + 11173962825 * k7n / 925320556 - 13158990841 * k8n / 6184727034 + 3936647629 * k9n / 1978049680 - 160528059 * k10n / 685178525 + 248638103 * k11n / 1413531060;
	
	h1 = _h + _timestep * (403863854 * k1h / 491063109 - 5068492393 * k4h / 434740067 - 411421997 * k5h / 543043805 + 652783627 * k6h / 914296604 + 11173962825 * k7h / 925320556 - 13158990841 * k8h / 6184727034 + 3936647629 * k9h / 1978049680 - 160528059 * k10h / 685178525 + 248638103 * k11h / 1413531060;
	
	w1 = _w + _timestep * (403863854 * k1w / 491063109 - 5068492393 * k4w / 434740067 - 411421997 * k5w / 543043805 + 652783627 * k6w / 914296604 + 11173962825 * k7w / 925320556 - 13158990841 * k8w / 6184727034 + 3936647629 * k9w / 1978049680 - 160528059 * k10w / 685178525 + 248638103 * k11w / 1413531060;

	v = _voltage + _timestep * (403863854 * k1V / 491063109 - 5068492393 * k4V / 434740067 - 411421997 * k5V / 543043805 + 652783627 * k6V / 914296604 + 11173962825 * k7V / 925320556 - 13158990841 * k8V / 6184727034 + 3936647629 * k9V / 1978049680 - 160528059 * k10V / 685178525 + 248638103 * k11V / 1413531060;
	
	t = _time + _timestep;
	n4 = n1 * n1 * n1 * n1;
	m3 = m1 * m1 * m1;

	k13V = kV(v, t, h1, w1, m3, n4);
	k13n = kn(v, n1);
	k13m = km(v, m1);
	k13h = kh(v, h1);
	k13w = kw(v, w1);
	*/
	//	update all values for next time point

	_voltage = _voltage + _timestep * (13451932 * k1V / 455176623 - 808719846 * k6V / 976000145 + 1757004468 * k7V / 5645159321 + 656045339 * k8V / 265891186 - 3867574721 * k9V / 1518517206 + 465885868 * k10V / 322736535 + 53011238 * k11V / 667516719 + 2 * k12V / 45);
	_n = _n + _timestep * (13451932 * k1n / 455176623 - 808719846 * k6n / 976000145 + 1757004468 * k7n / 5645159321 + 656045339 * k8n / 265891186 - 3867574721 * k9n / 1518517206 + 465885868 * k10n / 322736535 + 53011238 * k11n / 667516719 + 2 * k12n / 45);
	_m = _m + _timestep * (13451932 * k1m / 455176623 - 808719846 * k6m / 976000145 + 1757004468 * k7m / 5645159321 + 656045339 * k8m / 265891186 - 3867574721 * k9m / 1518517206 + 465885868 * k10m / 322736535 + 53011238 * k11m / 667516719 + 2 * k12m / 45);
	_h = _h + _timestep * (13451932 * k1h / 455176623 - 808719846 * k6h / 976000145 + 1757004468 * k7h / 5645159321 + 656045339 * k8h / 265891186 - 3867574721 * k9h / 1518517206 + 465885868 * k10h / 322736535 + 53011238 * k11h / 667516719 + 2 * k12h / 45);
	_w = _w + _timestep * (13451932 * k1w / 455176623 - 808719846 * k6w / 976000145 + 1757004468 * k7w / 5645159321 + 656045339 * k8w / 265891186 - 3867574721 * k9w / 1518517206 + 465885868 * k10w / 322736535 + 53011238 * k11w / 667516719 + 2 * k12w / 45);
	
	_Gexc = Ge(_time + _timestep);
	_Ginh = Gi(_time + _timestep);
	
	_time = _time + _timestep;
	_I = I_total(_time);
	

	// calculate errors
	/*
	double vBetter, vError;


	vBetter = _voltage + _timestep * (14005451 * k1V / 335480064 - 59238493 * k6V / 1068277825 + 181606767 * k7V / 758867731 + 561292985 * k8V / 797845732 - 1041891430 * k9V / 1371343529 + 760417239 * k10V / 1151165299 + 118820643 * k11V / 751138087 - 528747749 * k12V / 2220607170 + k13V / 4);

	vError = fabs(voltage[itime + 1] - vBetter);

	std::cout << "Error is " << vError << std::endl;
	*/
	//	update internal time

}

void HHI::raiseE(double G)
{
	_Gexc += G;
}

void HHI::raiseI(double G)
{
	_Ginh += G;
}

double HHI::I_total(double t)
{
	if (_injected_current)
		return _Iinjected(t);
	else
		return I_default(t);
}

void HHI::write_buffers()
{
	std::ofstream output;

	output.open(_filename, std::ios::out | std::ios::binary |  std::ofstream::app); //	open file to write binary data

	for (int i = 0; i < BUFFER_SIZE; i++)
	{
		output.write(reinterpret_cast<const char*>(&buffer_time[i]), sizeof(buffer_time[i]));
		output.write(reinterpret_cast<const char*>(&buffer_V[i]), sizeof(buffer_V[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Ge[i]), sizeof(buffer_Ge[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Gi[i]), sizeof(buffer_Gi[i]));
	}
	
	output.close();
	
}


double HHI::Ge(double t){return _Gexc * exp(-(t - _time) / tExc);}
double HHI::Gi(double t){return _Ginh * exp(-(t - _time) / tInh);}
double HHI::kV(double v, double t, double h, double w, double m3, double n4){
	return (-gL * (v - El) - gNa * h * m3 * (v - Ena) - gKdr * n4 * (v - Ek)
		- gKHT * w * (v - Ek) - Ge(t) * v - Gi(t) * (v - Ei) + 100000 * I_total(t) / A) / cm;}
double HHI::kn(double v, double n){return HHI::an(v)*(1 - n) - HHI::bn(v)*n;}
double HHI::km(double v, double m){return HHI::am(v)*(1 - m) - HHI::bm(v)*m;}
double HHI::kh(double v, double h){return HHI::ah(v)*(1 - h) - HHI::bh(v)*h;}
double HHI::kw(double v, double w){return (HHI::wInf(v) - w) / HHI::tauW(v);}


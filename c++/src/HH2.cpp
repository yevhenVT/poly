#include "poisson_noise.h"
#include "HH2.h"
#include "exception.h"
#include <iostream>
#include <algorithm>
#include <fstream>

using namespace std::placeholders;

const int HH2::BUFFER_SIZE = 1000; // size of the buffer 

// noise conductances
const double HH2::Gs_noise_inh = 0.035;
const double HH2::Gd_noise_inh = 0.045;
const double HH2::Gs_noise_exc = 0.035;
const double HH2::Gd_noise_exc = 0.045;

// noise spike frequencies
const double HH2::lambda_exc = 100.0;
const double HH2::lambda_inh = 100.0;

// neuron model parameters
const double HH2::cm_s = 1.0; // capacitance of somatic membrane in micro Farad / cm^2
const double HH2::Rc = 130; // original 55; myNeuronModel 130; newModel 200 ; now 1; coupling between dendrite and soma in MOhms 
const double HH2::As = 0.5; // soma surface area original: 5000 in micro meters squared, now: 0.5 in 10^-4 cm^2
const double HH2::GsL = 0.05; // original 0.1; myNeuronModel 0.05; newModel 0.1
const double HH2::GsNa = 60; // original 60; newModel 80; 
const double HH2::GsK = 8; // original 8; newModel 4;
const double HH2::EsL = -80; // original -80
const double HH2::EsNa = 55;
const double HH2::EsK = -90;
const double HH2::Ad = 1; // soma surface area original: 10000 in micro meters squared, now: 1 in 10^-4 cm^2
const double HH2::GdL = 0.1; // original 0.1 
const double HH2::GdCa = 55; // original 55; newModel 40
const double HH2::GdCaK = 150; // original 150 
const double HH2::EdL = -80; // original -80
const double HH2::EdCa = 120;
const double HH2::EdK = -90;
const double HH2::tExc = 5;
const double HH2::tInh = 5;

// thresholds
const double HH2::threshold_spike = 0;
const double HH2::threshold_burst = 0;
const double HH2::spike_margin = 5.0;
const double HH2::burst_margin = 5.0;


HH2::HH2() : _mu_soma(0.0), _sigma_soma(0.0), _mu_dend(0.0), _sigma_dend(0.0), cm_d(1.0), point_distribution{sqrt(6), -sqrt(6), 1, 1, 1, 1, 1, 1, 1, 1, 1, 
																										  -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,
																										  0, 0, 0, 0, 0, 0, 0}
{
	//internal state
	_fired_soma = false;
	_fired_dend = false;

	// noise
	_poisson_noise = false;

	_training_dend = false;
	_training_soma = false;

	_timestep = 0.0;

    _generator = nullptr;
    
    _recorded = false;
    _recorded_full = false;
    
    _buffer_counter = 0;
    
    // set initial values
	this->set_to_rest();
}

void HH2::set_soma_current(DDfunction fs)
{
	_training_soma = true;
	_Is_training = fs;
}

void HH2::set_dend_current(DDfunction fd)
{
	_training_dend = true;
	_Id_training = fd;
}

void HH2::set_white_noise(double mu_s, double sigma_s, double mu_d, double sigma_d)
{
    try
    {
        if (_generator == nullptr)
            throw NoGenerator("Noise generator is not set up!\n");
        else
        {
            _mu_soma = mu_s;
            _sigma_soma = sigma_s;
            _mu_dend = mu_d;
            _sigma_dend = sigma_d;
        }
    }

    catch (NoGenerator const& e)
    {
        std::cerr << "NoGenerator Exception: " << e.what() << std::endl;

    }

}

void HH2::set_no_white_noise()
{
    _mu_soma = 0.0;
	_sigma_soma = 0.0;
    _mu_dend = 0.0;
	_sigma_dend = 0.0;
}

void HH2::set_poisson_noise()
{
    _poisson_noise = true;
    
    this->initialize_poisson_noise(_noise_exc_soma, lambda_exc);
	this->initialize_poisson_noise(_noise_inh_soma, lambda_inh);
	this->initialize_poisson_noise(_noise_exc_dend, lambda_exc);
	this->initialize_poisson_noise(_noise_inh_dend, lambda_inh);
}

void HH2::set_no_poisson_noise(){_poisson_noise = false;}

void HH2::set_norecording()
{
	_recorded = false;
	_recorded_full = false;
}

void HH2::set_recording(std::string filename)
{
	_recorded = true;
	_filename = filename;
	
	buffer_time.resize(BUFFER_SIZE);
	buffer_Vs.resize(BUFFER_SIZE);
	buffer_Vd.resize(BUFFER_SIZE);
	buffer_Ged.resize(BUFFER_SIZE);
	buffer_Gid.resize(BUFFER_SIZE);
	buffer_Is.resize(BUFFER_SIZE);
	buffer_Id.resize(BUFFER_SIZE);
}


void HH2::set_recording_full(std::string filename)
{
	_recorded_full = true;
	_filename = filename;
	
	buffer_time.resize(BUFFER_SIZE);
	buffer_Vs.resize(BUFFER_SIZE);
	buffer_Vd.resize(BUFFER_SIZE);
	buffer_Ged.resize(BUFFER_SIZE);
	buffer_Gid.resize(BUFFER_SIZE);
	
	buffer_n.resize(BUFFER_SIZE);
	buffer_h.resize(BUFFER_SIZE);
	buffer_r.resize(BUFFER_SIZE);
	buffer_c.resize(BUFFER_SIZE);
	buffer_Ca.resize(BUFFER_SIZE);
}

void HH2::initialize_poisson_noise(double& noise_time, double lambda)
{
    try
    {
        if (_generator == nullptr)
            throw NoGenerator("Noise generator is not set up!\n");
        else
	        noise_time = 1000.0 * _generator->get_spike_time(lambda);
        
    }

    catch (NoGenerator const& e)
    {
        std::cerr << "NoGenerator Exception: " << e.what() << std::endl;

    }

}

void HH2::set_Ei(double E){_Egaba = E;}

void HH2::set_dynamics(double tS){_timestep = tS;}

void HH2::reset_time()
{
	_time = 0.0;
	
	_Is = IsExt(_time);
	_Id = IdExt(_time);
	
	//	initialize up noise
    if (_poisson_noise)
    {
	    this->initialize_poisson_noise(_noise_exc_soma, lambda_exc);
	    this->initialize_poisson_noise(_noise_inh_soma, lambda_inh);
	    this->initialize_poisson_noise(_noise_exc_dend, lambda_exc);
	    this->initialize_poisson_noise(_noise_inh_dend, lambda_inh);
    }
}

void HH2::set_to_rest()
{
	_time = 0.0;
	_Vs = -79.97619025;
	_Vd = -79.97268759;
	_n = 0.01101284;
	_h = 0.9932845;
	_r = 0.00055429;
	_c = 0.00000261762353;
	_Ca = 0.01689572;
	_Ginh_s = 0.0;
	_Gexc_s = 0.0;
	_Ginh_d = 0.0;
	_Gexc_d = 0.0;
	_Egaba = -80.0;

	_flag_soma = 0;
 	_flag_dend = 0;
 	
 	_Nspikes = 0;
 	_Nbursts = 0;

	_Is = IsExt(_time);
	_Id = IdExt(_time);
	
	_buffer_counter = 0;

	//	initialize up noise
    if (_poisson_noise)
    {
	    this->initialize_poisson_noise(_noise_exc_soma, lambda_exc);
	    this->initialize_poisson_noise(_noise_inh_soma, lambda_inh);
	    this->initialize_poisson_noise(_noise_exc_dend, lambda_exc);
	    this->initialize_poisson_noise(_noise_inh_dend, lambda_inh);
    }

}

void HH2::set_noise_generator(Poisson_noise* g)
{
	_generator = g;
}

double HH2::IdExt(double t)
{
	if (_training_dend)
		return _Id_training(t);
	else
		return Id_default(t);
}

double HH2::IsExt(double t)
{
	if (_training_soma)
		return _Is_training(t);
	else
		return Is_default(t);
}

bool HH2::get_fired_soma()
{
	return _fired_soma;
}

bool HH2::get_fired_dend()
{
	return _fired_dend;
}

int HH2::get_spike_number_soma()
{
    return _Nspikes;
}

int HH2::get_spike_number_dend()
{
    return _Nbursts;

}

void HH2::print_param()
{
	std::cout << "cm_s = " << cm_s << std::endl;
	std::cout << "cm_d = " << cm_d << std::endl;
	std::cout << "Rc = " << Rc << std::endl;
	std::cout << "As = " << As << std::endl;
	std::cout << "GsL = " << GsL << std::endl;
	std::cout << "GsNa = " << GsNa << std::endl;
	std::cout << "GsK = " << GsK << std::endl;
	std::cout << "EsL = " << EsL << std::endl;
	std::cout << "EsNa = " << EsNa << std::endl;
	std::cout << "EsK = " << EsK << std::endl;
	std::cout << "Ad = " << Ad << std::endl;
	std::cout << "GdL = " << GdL << std::endl;
	std::cout << "GdCa = " << GdCa << std::endl;
	std::cout << "GdCaK = " << GdCaK << std::endl;
	std::cout << "EdL = " << EdL << std::endl;
	std::cout << "EdCa = " << EdCa << std::endl;
	std::cout << "EdK = " << EdK << std::endl;
}

void HH2::write_buffers()
{
	std::ofstream output;

	output.open(_filename, std::ios::out | std::ios::binary |  std::ofstream::app); //	open file to write binary data

	for (int i = 0; i < BUFFER_SIZE; i++)
	{
		output.write(reinterpret_cast<const char*>(&buffer_time[i]), sizeof(buffer_time[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Vs[i]), sizeof(buffer_Vs[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Vd[i]), sizeof(buffer_Vd[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Ged[i]), sizeof(buffer_Ged[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Gid[i]), sizeof(buffer_Gid[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Is[i]), sizeof(buffer_Is[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Id[i]), sizeof(buffer_Id[i]));
	}
	
	output.close();
	
}


void HH2::write_buffers_full()
{
	std::ofstream output;

	output.open(_filename, std::ios::out | std::ios::binary |  std::ofstream::app); //	open file to write binary data

	for (int i = 0; i < BUFFER_SIZE; i++)
	{
		output.write(reinterpret_cast<const char*>(&buffer_time[i]), sizeof(buffer_time[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Vs[i]), sizeof(buffer_Vs[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Vd[i]), sizeof(buffer_Vd[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Ged[i]), sizeof(buffer_Ged[i]));
		output.write(reinterpret_cast<const char*>(&buffer_Gid[i]), sizeof(buffer_Gid[i]));
		
		output.write(reinterpret_cast<const char*>(&buffer_n[i]), sizeof(double));
		output.write(reinterpret_cast<const char*>(&buffer_h[i]), sizeof(double));
		output.write(reinterpret_cast<const char*>(&buffer_r[i]), sizeof(double));
		output.write(reinterpret_cast<const char*>(&buffer_c[i]), sizeof(double));
		output.write(reinterpret_cast<const char*>(&buffer_Ca[i]), sizeof(double));
	}
	
	output.close();
}

void HH2::raiseE(double G)
{
    _Gexc_d += G;
}

void HH2::raiseI(double G)
{
	_Ginh_d += G;
}

void HH2::record_state()
{		
	buffer_time[_buffer_counter] = _time;
	buffer_Vs[_buffer_counter] = _Vs;
	buffer_Vd[_buffer_counter] = _Vd;
	buffer_Ged[_buffer_counter] = _Gexc_d;
	buffer_Gid[_buffer_counter] = _Ginh_d;
	buffer_Is[_buffer_counter] = _Is;
	buffer_Id[_buffer_counter] = _Id;
	
	
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


void HH2::record_state_full()
{		
	buffer_time[_buffer_counter] = _time;
	buffer_Vs[_buffer_counter] = _Vs;
	buffer_Vd[_buffer_counter] = _Vd;
	buffer_Ged[_buffer_counter] = _Gexc_d;
	buffer_Gid[_buffer_counter] = _Ginh_d;
	
	buffer_n[_buffer_counter] = _n;
	buffer_h[_buffer_counter] = _h;
	buffer_r[_buffer_counter] = _r;
	buffer_c[_buffer_counter] = _c;
	buffer_Ca[_buffer_counter] = _Ca;
	
	
	_buffer_counter += 1;
	
	// if buffer is full, write it to the file
	if (_buffer_counter == BUFFER_SIZE)
	{
		this->write_buffers_full();
		_buffer_counter = 0;
	}
}

void HH2::R4_step_no_target_update()
{
	if (_recorded)
		this->record_state();
		
	this->state_noise_check();
	this->Runge4_step();
}
void HH2::Euler_Maruyama_step_no_target_update()
{
	if (_recorded)
		this->record_state();
	
	if (_recorded_full)
		this->record_state_full();
	
	this->state_noise_check();
	this->Euler_Maruyama_step();
}

void HH2::Debraband_step_no_target_update()
{
	if (_recorded)
		this->record_state();
	
	if (_recorded_full)
		this->record_state_full();
	
	this->state_noise_check();
	this->Debraband_step();
}


void HH2::state_noise_check()
{
	this->state_check();

	if (_poisson_noise)
	{
		this->noise_check(_Ginh_s, Gs_noise_inh, lambda_inh, _noise_inh_soma);
		this->noise_check(_Ginh_d, Gd_noise_inh, lambda_inh, _noise_inh_dend);
		this->noise_check(_Gexc_s, Gs_noise_exc, lambda_exc, _noise_exc_soma);
		this->noise_check(_Gexc_d, Gd_noise_exc, lambda_exc, _noise_exc_dend);
	}
}

void HH2::state_check()
{
    // somatic spike is defined as membrane potential second crossing of th threshold (when potential gows down)
	if ( (_flag_soma == 1) && (_Vs < threshold_spike - spike_margin) )
	{
		_flag_soma = 0;
		_fired_soma = true;
		_Nspikes++;
	}
	else
	{
		_fired_soma = false;
		
		//	check if we should change the state of neuron (voltage crossed the threshold)
		if ( (_flag_soma == 0) && (_Vs > threshold_spike) )
			_flag_soma = 1;
	}

    // dendritic spike is defined as membrane potential first crossing of the threshold (when potential gows up)
	if ( (_flag_dend == 0) && (_Vd >= threshold_burst) )
	{
		_flag_dend = 1;
		_fired_dend = true;
		_Nbursts++;
	}
	else
	{
		_fired_dend = false;
		
		//	check if we should change the state of neuron (voltage crossed the threshold)
		if ( (_flag_dend == 1) && (_Vd < threshold_burst - burst_margin))
			_flag_dend = 0;
	}
}

void HH2::noise_check(double& G, double G_noise, double lambda, double& noise_time)
{
	if (_time > noise_time)
		{
			G += _generator->random(G_noise);

			double random = 1000.0 * _generator->get_spike_time(lambda);
			
			while (random < _timestep)
			{
					random = 1000.0 * _generator->get_spike_time(lambda);
					G += _generator->random(G_noise);
			}
			noise_time = noise_time + random;
		}
}

void HH2::Euler_Maruyama_step()
{
	double Js = _generator->normal_distribution() * _sigma_soma;
	double Jd = _generator->normal_distribution() * _sigma_dend;

	_Vs = _Vs + _timestep * kVs(_Vs, _Vd, _n, _h, _time) + Js * sqrt(_timestep) * 100000 / (As * cm_s);
	_Vd = _Vd + _timestep * kVd(_Vs, _Vd, _r, _c, _Ca, _time) + Jd * sqrt(_timestep) * 100000 / (Ad * cm_d);

	_n = _n + _timestep * kn(_Vs, _n);
	_h = _h + _timestep * kh(_Vs, _h);
	_r = _r + _timestep * kr(_Vd, _r);
	_Ca = _Ca + _timestep * kCa(_Vd, _r, _Ca);
	_c = _c + _timestep * kc(_Vd, _c);
	
	_Gexc_d = Ge_d(_time + _timestep);
    _Gexc_s = Ge_s(_time + _timestep);

	_Ginh_d = Gi_d(_time + _timestep);
	_Ginh_s = Gi_s(_time + _timestep);
	
	_time = _time + _timestep;
	
	_Id = IdExt(_time) + _mu_dend + Jd;
	_Is = IsExt(_time) + _mu_soma + Js;
}

void HH2::Debraband_step()
{
	double H1Vs, H2Vs, H3Vs, H4Vs;	
	double H1Vd, H2Vd, H3Vd, H4Vd;	
	double H1Ca, H2Ca, H3Ca, H4Ca;	
	double H1n, H2n, H3n, H4n;	
	double H1h, H2h, H3h, H4h;	
	double H1c, H2c, H3c, H4c;	
	double H1r, H2r, H3r, H4r;	

	double a1Vs, a2Vs, a3Vs, a4Vs;	
	double a1Vd, a2Vd, a3Vd, a4Vd;	
	double a1Ca, a2Ca, a3Ca, a4Ca;	
	double a1n, a2n, a3n, a4n;	
	double a1h, a2h, a3h, a4h;	
	double a1c, a2c, a3c, a4c;	
	double a1r, a2r, a3r, a4r;	
	
	//double J1s = generator->normal_distribution();
	//double J2s = generator->normal_distribution();
	//double J1d = generator->normal_distribution();
	//double J2d = generator->normal_distribution();

	int ind1s, ind2s, ind1d, ind2d;

	ind1s = _generator->sample_index_for_point_distribution();
	ind2s = _generator->sample_index_for_point_distribution();
	ind1d = _generator->sample_index_for_point_distribution();
	ind2d = _generator->sample_index_for_point_distribution();

	double J1s = point_distribution[ind1s];
	double J2s = point_distribution[ind2s];
	double J1d = point_distribution[ind1d];
	double J2d = point_distribution[ind2d];
	
	// with A in micro meters ^2
	//H1Vs = _Vs + sqrt(_timestep) * _sigma_soma * (-0.01844540496323970 * J1s - 0.1866426386543421 * J2s) * 100000 / (As * cm_s);
	//H1Vd = _Vd + sqrt(_timestep) * _sigma_dend * (-0.01844540496323970 * J1d - 0.1866426386543421 * J2d) * 100000 / (Ad * cm_d);
	
	// with A in 10^-4 cm^2
	H1Vs = _Vs + sqrt(_timestep) * _sigma_soma * (-0.01844540496323970 * J1s - 0.1866426386543421 * J2s) * 10 / (As * cm_s);
	H1Vd = _Vd + sqrt(_timestep) * _sigma_dend * (-0.01844540496323970 * J1d - 0.1866426386543421 * J2d) * 10 / (Ad * cm_d);

	H1Ca = _Ca;
	H1n  = _n;
	H1h  = _h;
	H1c  = _c;
	H1r  = _r;

	a1Vs = kVs(H1Vs, H1Vd, H1n, H1h, _time);	
	a1Vd = kVd(H1Vs, H1Vd, H1r, H1c, H1Ca, _time);
	a1n = kn(H1Vs, H1n);
	a1h = kh(H1Vs, H1h);
	a1r = kr(H1Vd, H1r);
	a1c = kc(H1Vd, H1c);
	a1Ca = kCa(H1Vd, H1r, H1Ca);

	// with A in micro meters ^2
	//H2Vs = _Vs + _timestep * a1Vs + sqrt(_timestep) * _sigma_soma * (0.8017012756521233 * J1s - 0.8575745885712401 * J2s) * 100000 / (As * cm_s);
	//H2Vd = _Vd + _timestep * a1Vd + sqrt(_timestep) * _sigma_dend * (0.8017012756521233 * J1d - 0.8575745885712401 * J2d) * 100000 / (Ad * cm_d);
	
	// with A in 10^-4 cm^2
	H2Vs = _Vs + _timestep * a1Vs + sqrt(_timestep) * _sigma_soma * (0.8017012756521233 * J1s - 0.8575745885712401 * J2s) * 10 / (As * cm_s);
	H2Vd = _Vd + _timestep * a1Vd + sqrt(_timestep) * _sigma_dend * (0.8017012756521233 * J1d - 0.8575745885712401 * J2d) * 10 / (Ad * cm_d);
	
	H2Ca = _Ca + _timestep * a1Ca;
	H2n =  _n  + _timestep * a1n;
	H2h =  _h  + _timestep * a1h;
	H2c =  _c  + _timestep * a1c;
	H2r =  _r  + _timestep * a1r;

	a2Vs = kVs(H2Vs, H2Vd, H2n, H2h, _time + _timestep);	
	a2Vd = kVd(H2Vs, H2Vd, H2r, H2c, H2Ca, _time + _timestep);
	a2n = kn(H2Vs, H2n);
	a2h = kh(H2Vs, H2h);
	a2r = kr(H2Vd, H2r);
	a2c = kc(H2Vd, H2c);
	a2Ca = kCa(H2Vd, H2r, H2Ca);

	// with A in micro meters ^2
	//H3Vs = _Vs + _timestep * (3 * a1Vs + a2Vs) / 8 + sqrt(_timestep) * _sigma_soma * (0.5092227024816198 * J1s - 0.4723392695015512 * J2s) * 100000 / (As * cm_s);
	//H3Vd = _Vd + _timestep * (3 * a1Vd + a2Vd) / 8 + sqrt(_timestep) * _sigma_dend * (0.5092227024816198 * J1d - 0.4723392695015512 * J2d) * 100000 / (Ad * cm_d);
	
	// with A in 10^-4 cm^2
	H3Vs = _Vs + _timestep * (3 * a1Vs + a2Vs) / 8 + sqrt(_timestep) * _sigma_soma * (0.5092227024816198 * J1s - 0.4723392695015512 * J2s) * 10 / (As * cm_s);
	H3Vd = _Vd + _timestep * (3 * a1Vd + a2Vd) / 8 + sqrt(_timestep) * _sigma_dend * (0.5092227024816198 * J1d - 0.4723392695015512 * J2d) * 10 / (Ad * cm_d);
	
	H3Ca = _Ca + _timestep * (3 * a1Ca + a2Ca) / 8;
	H3n  = _n  + _timestep * (3 * a1n + a2n) / 8;
	H3h  = _h  + _timestep * (3 * a1h + a2h) / 8;
	H3c  = _c  + _timestep * (3 * a1c + a2c) / 8;
	H3r  = _r  + _timestep * (3 * a1r + a2r) / 8;

	a3Vs = kVs(H3Vs, H3Vd, H3n, H3h, _time + 0.5 * _timestep);	
	a3Vd = kVd(H3Vs, H3Vd, H3r, H3c, H3Ca, _time + 0.5 * _timestep);
	a3n = kn(H3Vs, H3n);
	a3h = kh(H3Vs, H3h);
	a3r = kr(H3Vd, H3r);
	a3c = kc(H3Vd, H3c);
	a3Ca = kCa(H3Vd, H3r, H3Ca);

	// with A in micro meters ^2
	//H4Vs = _Vs + _timestep * (-0.4526683126055039 * a1Vs - 0.4842227708685013 * a2Vs + 1.9368910834740051 * a3Vs)
	//	 + sqrt(_timestep) * _sigma_soma * (0.9758794209767762 * J1s + 0.3060354860326548 * J2s) * 100000 / (As * cm_s);
	
	//H4Vd = _Vd + _timestep * (-0.4526683126055039 * a1Vd - 0.4842227708685013 * a2Vd + 1.9368910834740051 * a3Vd)
	//	 + sqrt(_timestep) * _sigma_dend * (0.9758794209767762 * J1d + 0.3060354860326548 * J2d) * 100000 / (Ad * cm_d);

	// with A in 10^-4 cm^2
	H4Vs = _Vs + _timestep * (-0.4526683126055039 * a1Vs - 0.4842227708685013 * a2Vs + 1.9368910834740051 * a3Vs)
		 + sqrt(_timestep) * _sigma_soma * (0.9758794209767762 * J1s + 0.3060354860326548 * J2s) * 10 / (As * cm_s);
	
	H4Vd = _Vd + _timestep * (-0.4526683126055039 * a1Vd - 0.4842227708685013 * a2Vd + 1.9368910834740051 * a3Vd)
		 + sqrt(_timestep) * _sigma_dend * (0.9758794209767762 * J1d + 0.3060354860326548 * J2d) * 10 / (Ad * cm_d);


	H4Ca = _Ca + _timestep * (-0.4526683126055039 * a1Ca - 0.4842227708685013 * a2Ca + 1.9368910834740051 * a3Ca);
	H4n  = _n  + _timestep * (-0.4526683126055039 * a1n - 0.4842227708685013 * a2n + 1.9368910834740051 * a3n);
	H4h  = _h  + _timestep * (-0.4526683126055039 * a1h - 0.4842227708685013 * a2h + 1.9368910834740051 * a3h);
	H4c  = _c  + _timestep * (-0.4526683126055039 * a1c - 0.4842227708685013 * a2c + 1.9368910834740051 * a3c);
	H4r  = _r  + _timestep * (-0.4526683126055039 * a1r - 0.4842227708685013 * a2r + 1.9368910834740051 * a3r);

	a4Vs = kVs(H4Vs, H4Vd, H4n, H4h, _time + _timestep);	
	a4Vd = kVd(H4Vs, H4Vd, H4r, H4c, H4Ca, _time + _timestep);
	a4n = kn(H4Vs, H4n);
	a4h = kh(H4Vs, H4h);
	a4r = kr(H4Vd, H4r);
	a4c = kc(H4Vd, H4c);
	a4Ca = kCa(H4Vd, H4r, H4Ca);

	// with A in micro meters ^2
	//_Vs = _Vs + _timestep * (a1Vs / 6 - 0.005430430675258792 * a2Vs + 2 * a3Vs / 3 + 0.1720970973419255 * a4Vs)
	//			  + sqrt(_timestep) * _sigma_soma * J1s  * 100000 / (As * cm_s);
	
	//_Vd = _Vd + _timestep * (a1Vd / 6 - 0.005430430675258792 * a2Vd + 2 * a3Vd / 3 + 0.1720970973419255 * a4Vd)
	//			  + sqrt(_timestep) * _sigma_dend * J1d * 100000 / (Ad * cm_d);
	
	// with A in 10^-4 cm^2
	_Vs = _Vs + _timestep * (a1Vs / 6 - 0.005430430675258792 * a2Vs + 2 * a3Vs / 3 + 0.1720970973419255 * a4Vs)
				  + sqrt(_timestep) * _sigma_soma * J1s  * 10 / (As * cm_s);
	
	_Vd = _Vd + _timestep * (a1Vd / 6 - 0.005430430675258792 * a2Vd + 2 * a3Vd / 3 + 0.1720970973419255 * a4Vd)
				  + sqrt(_timestep) * _sigma_dend * J1d * 10 / (Ad * cm_d);
	
	_Ca = _Ca + _timestep * (a1Ca / 6 - 0.005430430675258792 * a2Ca + 2 * a3Ca / 3 + 0.1720970973419255 * a4Ca);
	_n  = _n  + _timestep * (a1n / 6 - 0.005430430675258792 * a2n + 2 * a3n / 3 + 0.1720970973419255 * a4n);
	_h  = _h  + _timestep * (a1h / 6 - 0.005430430675258792 * a2h + 2 * a3h / 3 + 0.1720970973419255 * a4h);
	_r  = _r  + _timestep * (a1r / 6 - 0.005430430675258792 * a2r + 2 * a3r / 3 + 0.1720970973419255 * a4r);
	_c  = _c  + _timestep * (a1c / 6 - 0.005430430675258792 * a2c + 2 * a3c / 3 + 0.1720970973419255 * a4c);

	_Gexc_d = Ge_d(_time + _timestep);
    _Gexc_s = Ge_s(_time + _timestep);

	_Ginh_d = Gi_d(_time + _timestep);
	_Ginh_s = Gi_s(_time + _timestep);
	
	_time = _time + _timestep;
	
	_Id = IdExt(_time);
	_Is = IsExt(_time);
	
	_Id += _mu_dend + J1d * _sigma_dend;
	_Is += _mu_soma + J1s * _sigma_soma;
}

void HH2::Runge4_step()
{
	double n1, h1, c1, r1;	//	temporary values of gating variables
	double vts, vtd, Cat;	//	temporary values of variables
	double k1Vs, k2Vs, k3Vs, k4Vs;
	double k1Vd, k2Vd, k3Vd, k4Vd;
	double k1Ca, k2Ca, k3Ca, k4Ca;
	double k1n, k2n, k3n, k4n;
	double k1h, k2h, k3h, k4h;
	double k1c, k2c, k3c, k4c;
	double k1r, k2r, k3r, k4r;
	double t;

	vts = _Vs;
	n1  = _n;
	h1  = _h;
	vtd = _Vd;
	r1  = _r;
	c1  = _c;
	Cat = _Ca;
	t   = _time;

	k1Vs = kVs(vts, vtd, n1, h1, t);
	k1n = kn(vts, n1);
	k1h = kh(vts, h1);
	k1Vd = kVd(vts, vtd, r1, c1, Cat, t);
	k1r = kr(vtd, r1);
	k1c = kc(vtd, c1);
	k1Ca = kCa(vtd, r1, Cat);

	vts = _Vs + _timestep * k1Vs / 3;
	n1  = _n + _timestep * k1n / 3;
	h1  = _h + _timestep * k1h / 3;
	vtd = _Vd + _timestep * k1Vd / 3;
	r1  = _r + _timestep * k1r / 3;
	c1  = _c + _timestep * k1c / 3;
	Cat = _Ca + _timestep * k1Ca / 3;

	t = _time + _timestep / 3;

	k2Vs = kVs(vts, vtd, n1, h1, t);
	k2n = kn(vts, n1);
	k2h = kh(vts, h1);
	k2Vd = kVd(vts, vtd, r1, c1, Cat, t);
	k2r = kr(vtd, r1);
	k2c = kc(vtd, c1);
	k2Ca = kCa(vtd, r1, Cat);

	vts = _Vs + _timestep * (-k1Vs / 3 + k2Vs);
	n1  = _n + _timestep * (-k1n / 3 + k2n);
	h1  = _h + _timestep * (-k1h / 3 + k2h);
	vtd = _Vd + _timestep * (-k1Vd / 3 + k2Vd);
	r1  = _r + _timestep * (-k1r / 3 + k2r);
	c1  = _c + _timestep * (-k1c / 3 + k2c);
	Cat = _Ca + _timestep * (-k1Ca / 3 + k2Ca);

	t = _time + 2 * _timestep / 3;

	k3Vs = kVs(vts, vtd, n1, h1, t);
	k3n = kn(vts, n1);
	k3h = kh(vts, h1);
	k3Vd = kVd(vts, vtd, r1, c1, Cat, t);
	k3r = kr(vtd, r1);
	k3c = kc(vtd, c1);
	k3Ca = kCa(vtd, r1, Cat);

	vts = _Vs + _timestep * (k1Vs - k2Vs + k3Vs);
	n1  = _n + _timestep * (k1n - k2n + k3n);
	h1  = _h + _timestep * (k1h - k2h + k3h);
	vtd = _Vd + _timestep * (k1Vd - k2Vd + k3Vd);
	r1  = _r + _timestep * (k1r - k2r + k3r);
	c1  = _c + _timestep * (k1c - k2c + k3c);
	Cat = _Ca + _timestep * (k1Ca - k2Ca + k3Ca);

	t = _time + _timestep;

	k4Vs = kVs(vts, vtd, n1, h1, t);
	k4n = kn(vts, n1);
	k4h = kh(vts, h1);
	k4Vd = kVd(vts, vtd, r1, c1, Cat, t);
	k4r = kr(vtd, r1);
	k4c = kc(vtd, c1);
	k4Ca = kCa(vtd, r1, Cat);

	_Gexc_d = Ge_d(_time + _timestep);
    _Gexc_s = Ge_s(_time + _timestep);

	_Ginh_d = Gi_d(_time + _timestep);
	_Ginh_s = Gi_s(_time + _timestep);
	
	_Vs = _Vs + _timestep * (k1Vs + 3 * k2Vs + 3 * k3Vs + k4Vs) / 8;
	_n = _n + _timestep * (k1n + 3 * k2n + 3 * k3n + k4n) / 8;
	_h = _h + _timestep * (k1h + 3 * k2h + 3 * k3h + k4h) / 8;
	_Vd = _Vd + _timestep * (k1Vd + 3 * k2Vd + 3 * k3Vd + k4Vd) / 8;
	_r = _r + _timestep * (k1r + 3 * k2r + 3 * k3r + k4r) / 8;
	_c = _c + _timestep * (k1c + 3 * k2c + 3 * k3c + k4c) / 8;
	_Ca = _Ca + _timestep * (k1Ca + 3 * k2Ca + 3 * k3Ca + k4Ca) / 8;
	
	_time = _time + _timestep;
	
	_Id = IdExt(_time);
	_Is = IsExt(_time);
}


double HH2::Gi_s(double t){return _Ginh_s * exp(-(t - _time) / tInh);}
double HH2::Gi_d(double t){return _Ginh_d * exp(-(t - _time) / tInh);}
double HH2::Ge_d(double t){return _Gexc_d * exp(-(t - _time) / tExc);}
double HH2::Ge_s(double t){return _Gexc_s * exp(-(t - _time) / tExc);}

double HH2::kVs(double vs, double vd, double n, double h, double t)
{
	double m3, n4;

	m3 = mInf(vs) * mInf(vs) * mInf(vs);
	n4 = n * n * n * n;
	// with A in micro meters ^2
	//return (-GsL * (vs - EsL) - GsNa * m3 * h * (vs - EsNa) - GsK * n4 * (vs - EsK)
	//	 - Gi_s(t) * (vs - _Egaba) - Ge_s(t) * vs + 100000 * (IsExt(t) + _mu_soma) / As + 100000 * (vd - vs) / (Rc * As)) / cm_s;
	
	// with A in 10^-4 cm^2
	return (-GsL * (vs - EsL) - GsNa * m3 * h * (vs - EsNa) - GsK * n4 * (vs - EsK)
		 - Gi_s(t) * (vs - _Egaba) - Ge_s(t) * vs + 10 * (IsExt(t) + _mu_soma) / As + 10 * (vd - vs) / (Rc * As)) / cm_s;
}
double HH2::kVd(double vs, double vd, double r, double c, double ca, double t)
{
	double r2;

	r2 = r * r;
	// with A in micro meters ^2
	//return (-GdL * (vd - EdL) - GdCa * r2 * (vd - EdCa) - GdCaK * (vd - EdK) * c / (1 + 6 / ca)
	//	- Ge_d(t) * vd - Gi_d(t) * (vd - _Egaba) + 100000 * (IdExt(t) + _mu_dend) / Ad + 100000 * (vs - vd) / (Rc * Ad)) / cm_d;
	
	// with A in 10^-4 cm^2
	return (-GdL * (vd - EdL) - GdCa * r2 * (vd - EdCa) - GdCaK * (vd - EdK) * c / (1 + 6 / ca)
		- Ge_d(t) * vd - Gi_d(t) * (vd - _Egaba) + 10 * (IdExt(t) + _mu_dend) / Ad + 10 * (vs - vd) / (Rc * Ad)) / cm_d;
}

double HH2::kn(double vs, double n){return (HH2::nInf(vs) - n) / HH2::tauN(vs);}
double HH2::kh(double vs, double h){return (HH2::hInf(vs) - h) / HH2::tauH(vs);}
double HH2::kr(double vd, double r){return (HH2::rInf(vd) - r) / HH2::tauR(vd);}
double HH2::kc(double vd, double c){return (HH2::cInf(vd) - c) / HH2::tauC(vd);}
double HH2::kCa(double vd, double r, double ca)
	{return (-0.1 * GdCa * r * r * (vd - EdCa) - 0.02 * ca);} // original -0.1; -0.02

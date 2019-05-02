The code was used to produce results of the paper "Local axonal conduction delays underlie precise timing of a neural sequence" (currently under review). In short, we suggest a new feedforward network architecture that is able to produce precise temporal sequences. We develop an algorithm to generate a network and illustrate how measured axonal conduction delays in songbird HVC naturally incorporate into the model and produce a uniform temporal activity. More details will be provided once the paper is published.

Author of the model: Yevhen Tupikov (2018-2019)
We provide the source code used to generate Figures 2-5. Since the process of wiring polychronous networks takes considerable time (up to 30 hours using 20 CPU cores), we only run the code for simulation of synfire chain dynamics and give examples on how to run the code for polychronous network. However, we provide all the data files with simulation results and the python scripts to visualize them.

REQUIREMENTS:
Below we list the library and utility versions that were tested. Other versions are also likely to work.
For c++ source code compilation:
Linux environment
GNU Make >=3.81 
GCC >=4.9.2 (other versions that support C++ 11 should work as well)
Openmpi >= 1.10.1

Note: After installing OpenMPI, please make sure that the path to the executables (mpirun, mpic++ etc.) is added to your environment PATH variable:

export PATH=/usr/lib64/mpi/gcc/openmpi/bin:$PATH

The exact path may vary depending on your installation.

For python visualizations:
Python 2.7
Matplotlib >= 2.02
Numpy >= 1.15.4

RUN:
The code supports two functionalities:
1.)	Wiring a synfire chain or a polychronous network
2.)	Testing wired network by running multiple simulations of dynamics
First, run the build.sh script in c++ folder. The script compiles executable files, creates a network with 20000 HVC-RA neurons, wires synfire chain with 170 neurons in each layer, simulates a single trial of dynamics, and visualizes an axonal conduction delay distribution and spike raster plot.
To wire a network, go to the folder createNetwork and see examples in the file examples.txt
To test a wired network, go to the folder testNetwork and see examples in the file examples.txt

VISUALIZE RESULTS FOR POLYCHRONOUS NETWORK:
To see the axonal conduction delays distributions and spike raster plots for grown polychronous networks, go to the folder python_scripts and run the following:
python show_results_polychronous.py -m <mean delay> -s <std delay> -d <time duration to plot>

where 
<mean delay> - mean of axonal conduction delays
<std delay>  - standard deviation of axonal conduction delays
<time duration to plot> - duration of the spike raster plot to show
 
Example that shows axonal conduction delay distribution and 100 ms of network dynamics for polychronous network with mean 2.5ms and std 1.25ms: 
python show_results_polychronous.py -m 2.5 -s 1.25 -d 100.0



#!/bin/sh
mkdir -p "synfireChain"
dataDir=$PWD/synfireChain
cd c++/createNetwork
printf "Build createNetwork\n"
make
printf "finished\n\n"
printf "Build testNetwork\n"
cd ../testNetwork
make
printf "finished\n\n"
printf "Wiring synfire chain with 170 neurons in each group and zero axonal conduction delays\n\n"
cd ..
mpirun -np 1 createNetwork/createNetwork -o $dataDir/ -nra 20000 -nout 170 -gee 0.004 -npsc 1 -md 0.0 -sd 0.0 --synfire
printf "\nSimulating synfire chain dynamics\n\n"
mpirun -np 20 testNetwork/testNetwork -n $dataDir/ -o $dataDir/ -f testTrial_ -nt 1 -d 800.0 -s 0.0 -gee 0.004 --resample
printf "\nPlot results\n\n"
printf "\nPlot cdf of axonal conduction delays\n\n"
python ../python_scripts/show_axonal_conduction_dist.py -f $dataDir/RA_RA_connections.bin
printf "\nPlot spike raster plot\n\n"
python ../python_scripts/show_spike_raster.py -f $dataDir/testTrial_0_somaSpikes.bin -d 150.0




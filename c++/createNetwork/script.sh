#!/bin/sh
#mpirun -np 1 ./createNetwork -o /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/nospace/random/ -nra 20000 -nout 170 -gee 0.02 -md 0.0 -sd 0.0 --random
#mpirun -np 1 ./createNetwork -o /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/nospace/synfireChain/ -nra 20000 -nout 170 -gee 0.004 -npsc 1 -md 3.4 -sd 2.27 --synfire
#mpirun -np 23 ./createNetwork -o /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/synfireChain/ -nra 20000 -ni 5500 -nout 170 -gee 0.004 -gei 0.0 -gie 0.0 -npsc 1  --synfire-space
#mpirun -np 23 ./createNetwork -o /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/nospace/poly/ -nra 20000 -nout 170 -maxnin 180 -gee 0.004 -sm 1.0 -md 3.4 -sd 2.27 --polychronous
mpirun -np 1 ./createNetwork -o /home/eugene/Programming/data/mlong/noise/network/ -nra 20000 -nout 170 -gee 0.004 -npsc 1 -md 0.0 -sd 0.0 --synfire


mpirun -np 40 ./createNetwork -o /home/eugene/Programming/data/mlong/integrationConst/poly/ -nra 20000 -nout 170 -maxnin 180 -gee 0.004 -sm 1.0 -md 3.4 -sd 2.27 --polychronous
mpirun -np 40 ./createNetwork -o /home/eugene/Programming/data/mlong/integrationConst/poly3/ -nra 20000 -nout 170 -maxnin 180 -gee 0.004 -sm 1.0 -c_min 1.0 -c_max 4.0 --integration_times
mpirun -np 40 ./createNetwork -o /home/eugene/Programming/data/mlong/integrationConst/poly6/ -nra 20000 -nout 170 -maxnin 180 -gee 0.004 -sm 6.0 -c_min 0.5 -c_mean 1.0 -c_std 0.01 --integration_times_lognormal
mpirun -np 40 ./createNetwork -o /home/eugene/Programming/data/mlong/integrationConst/poly8/ -nra 20000 -nout 170 -maxnin 180 -gee 0.004 -sm 6.0 -c_min 0.5 -c_mean 1.0 -c_std 0.4 --integration_times_lognormal


# in terms of integration time distribution
mpirun -np 40 ./createNetwork -o /home/eugene/Programming/data/mlong/integrationConst/poly13/ -nra 20000 -nout 170 -maxnin 180 -gee 0.004 -sm 6.0 -int_min 4.0 -int_max 20.0 -int_mean 5.5 -int_std 2.0 --integration_times_lognormal

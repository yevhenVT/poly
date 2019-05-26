#!/bin/sh
#mpirun -np 1 ./createNetwork -o /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/nospace/random/ -nra 20000 -nout 170 -gee 0.02 -md 0.0 -sd 0.0 --random
#mpirun -np 1 ./createNetwork -o /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/nospace/synfireChain/ -nra 20000 -nout 170 -gee 0.004 -npsc 1 -md 3.4 -sd 2.27 --synfire
#mpirun -np 23 ./createNetwork -o /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/synfireChain/ -nra 20000 -ni 5500 -nout 170 -gee 0.004 -gei 0.0 -gie 0.0 -npsc 1  --synfire-space
#mpirun -np 23 ./createNetwork -o /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/nospace/poly/ -nra 20000 -nout 170 -maxnin 180 -gee 0.004 -sm 1.0 -md 3.4 -sd 2.27 --polychronous
mpirun -np 1 ./createNetwork -o /home/eugene/Programming/data/mlong/noise/network/ -nra 20000 -nout 170 -gee 0.004 -npsc 1 -md 0.0 -sd 0.0 --synfire

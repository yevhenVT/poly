#!/bin/sh
#mpirun -np 23 ./testNetwork -n /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/synfireChainNoSpace/ -o /home/eugene/Output/networks/MLong/simResults/dendritic_tree/check/synfireChainNoSpace/ -f testTrial_ -nt 1 -d 300.0 -s 0.0

#mpirun -np 23 ./testNetwork -n /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/nospace/synfireChain/ -o /home/eugene/Output/networks/MLong/simResults/dendritic_tree/check/nospace/synfireChain/ -f testTrial_ -nt 3 -d 300.0 -s 0.0 -gee 0.004 --resample

#mpirun -np 23 ./testNetwork -n /home/eugene/Programming/data/mlong/uniform/grid10/ -o /home/eugene/Output/Programming/data/mlong/uniform/grid10/ -f testTrial_ -nt 50 -d 1000.0 -s 0.0
#mpirun -np 40 ./testNetwork -n /home/eugene/Programming/data/mlong/noise/network/ -o /home/eugene/Output/Programming/data/mlong/noise/noise_s0.1_d0.2/ -f testTrial_ -nt 50 -d 1000.0 -s 0.0
mpirun -np 40 ./testNetwork -n /home/eugene/Programming/data/mlong/noise/network/ -o /home/eugene/Programming/data/mlong/noise/052519/noise_s0.26_d0.0/ -f testTrial_ -nt 50 -d 1000.0 -s 0.0

#mpirun -np 23 ./testNetwork -n /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/synfireChain/ -o /home/eugene/Output/networks/MLong/simResults/dendritic_tree/check/synfireChain/ -f testTrial_ -nt 1 -d 300.0 -s 0.0 -gee 0.016 -gei 0.25 -gie 0.03 --resample-space

#mpirun -np 23 ./testNetwork -n /home/eugene/Output/networks/MLong/generateNetwork/dendritic_tree/check/nospace/random/ -o /home/eugene/Output/networks/MLong/simResults/dendritic_tree/check/nospace/random/ -f testTrial_ -nt 5 -d 200.0 -s 0.0 -gee 0.04 --resample --random


mpirun -np 40 ./testNetwork -n /home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/ -o /home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/test/ -f testTrial_ -nt 50 -d 1000.0 -s 0.0
mpirun -np 40 ./testNetwork -n /home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.2/ -o /home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.2/test/ -f testTrial_ -nt 50 -d 1300.0 -s 0.0

mpirun -np 40 ./testNetwork -n /home/eugene/Programming/data/mlong/integrationConst/grid/grid1/ -o /home/eugene/Programming/data/mlong/integrationConst/grid/grid1/test/ -f testTrial_ -nt 50 -d 1300.0 -s 0.0

mpirun -np 40 ./testNetwork -n /home/eugene/Programming/data/mlong/integrationConst/grid/grid5_seed1991/ -o /home/eugene/Programming/data/mlong/integrationConst/grid/grid5_seed1991/testTraces/ -f testTrial_ -nt 50 -d 900.0 -s 0.0 --record

mpirun -np 40 ./testNetwork -n /home/eugene/Programming/data/mlong/integrationConst/grid/grid5_seed1991/ -o /home/eugene/Programming/data/mlong/integrationConst/grid/grid5_seed1991/testTracesScale10.0/ -f testTrial_ -nt 50 -d 650.0 -s 0.0 --record

mpirun -np 40 ./testNetwork -n /home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/ -o /home/eugene/Programming/data/mlong/integrationConst/polyScaled/scaled2.0/ -f testTrial_ -nt 50 -d 1000.0 -s 0.0 --record

mpirun -np 40 ./testNetwork -n /home/eugene/Programming/data/mlong/integrationConst/gee0.032/poly3/ -o /home/eugene/Programming/data/mlong/integrationConst/gee0.032/poly3/test/ -f testTrial_ -nt 50 -d 400.0 -s 0.0 --record

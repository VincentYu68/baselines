export ENVNAME='--env DartHopper-v1'
export EXPNAME='--expname _torso_011_'
export BATCHSIZE=2000

if false; then
# session 1
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 1000 --split_percent 0.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 1000 --split_percent 0.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 3 --batch $BATCHSIZE --split_iter 1000 --split_percent 0.0
fi

if false; then
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 1 --split_percent 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 1 --split_percent 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 3 --batch $BATCHSIZE --split_iter 1 --split_percent 1.0
fi

if false; then
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 50 --split_percent 0.05
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 50 --split_percent 0.05
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 3 --batch $BATCHSIZE --split_iter 50 --split_percent 0.05
fi

if false; then
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 50 --split_percent 0.25
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 50 --split_percent 0.25
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 3 --batch $BATCHSIZE --split_iter 50 --split_percent 0.25
fi
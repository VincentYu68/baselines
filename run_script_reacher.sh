export ENVNAME='--env DartReacher3d-v1'
export EXPNAME='--expname _3modelsr05_diffaxis_new_jointlimit_revcont_append_'
export BATCHSIZE=1000

#if false; then
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 1000 --split_percent 0.0 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 1000 --split_percent 0.0 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 1 --split_percent 1.0 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 1 --split_percent 1.0 --ob_rms 0 --final_std 1.0
#fi

if false; then
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 100 --split_percent 0.05 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 100 --split_percent 0.05 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 100 --split_percent 0.25 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 100 --split_percent 0.25 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 100 --split_percent 0.5 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 100 --split_percent 0.5 --ob_rms 0 --final_std 1.0
fi

if false; then
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 200 --split_percent 0.05 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 200 --split_percent 0.05 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 200 --split_percent 0.25 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 200 --split_percent 0.25 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 200 --split_percent 0.5 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 200 --split_percent 0.5 --ob_rms 0 --final_std 1.0
fi

if false; then
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 20 --split_percent 0.05 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 20 --split_percent 0.05 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 20 --split_percent 0.25 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 20 --split_percent 0.25 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 20 --split_percent 0.5 --ob_rms 0 --final_std 1.0
#mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 20 --split_percent 0.5 --ob_rms 0 --final_std 1.0
fi
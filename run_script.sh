export ENVNAME='--env DartReacher3d-v1'
export EXPNAME='--expname _2task_updown_'
export BATCHSIZE=5000

if false; then
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 1000 --split_percent 0.0 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 1000 --split_percent 0.0 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 3 --batch $BATCHSIZE --split_iter 1000 --split_percent 0.0 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 1 --split_percent 1.0 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 1 --split_percent 1.0 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 1 --split_percent 1.0 --ob_rms 0 --final_std 1.0
fi

#if false; then
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 50 --split_percent 0.25 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 50 --split_percent 0.25 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 3 --batch $BATCHSIZE --split_iter 50 --split_percent 0.25 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 1 --batch $BATCHSIZE --split_iter 50 --split_percent 0.5 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 2 --batch $BATCHSIZE --split_iter 50 --split_percent 0.5 --ob_rms 0 --final_std 1.0
mpirun -np 4 python -m baselines.split_net.run_exp $ENVNAME $EXPNAME --seed 3 --batch $BATCHSIZE --split_iter 50 --split_percent 0.5 --ob_rms 0 --final_std 1.0
#fi



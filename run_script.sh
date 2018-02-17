if false; then
# session 1
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 1 --batch 2000 --split_iter 1000 --split_percent 0.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 2 --batch 2000 --split_iter 1000 --split_percent 0.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 3 --batch 2000 --split_iter 1000 --split_percent 0.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 1 --batch 2000 --split_iter 0 --split_percent 1.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 2 --batch 2000 --split_iter 0 --split_percent 1.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 3 --batch 2000 --split_iter 0 --split_percent 1.0

mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 1 --batch 2000 --split_iter 10 --split_percent 0.05 --split_interval 5
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 2 --batch 2000 --split_iter 10 --split_percent 0.05 --split_interval 5
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 3 --batch 2000 --split_iter 10 --split_percent 0.05 --split_interval 5
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 1 --batch 2000 --split_iter 10 --split_percent 0.0 --split_interval 5 --adapt_split 1
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 2 --batch 2000 --split_iter 10 --split_percent 0.0 --split_interval 5 --adapt_split 1
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfwdbwd_ --seed 3 --batch 2000 --split_iter 10 --split_percent 0.0 --split_interval 5 --adapt_split 1
fi

#if false; then
# session 2
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 1 --batch 2000 --split_iter 1000 --split_percent 0.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 2 --batch 2000 --split_iter 1000 --split_percent 0.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 3 --batch 2000 --split_iter 1000 --split_percent 0.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 1 --batch 2000 --split_iter 0 --split_percent 1.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 2 --batch 2000 --split_iter 0 --split_percent 1.0
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 3 --batch 2000 --split_iter 0 --split_percent 1.0

mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 1 --batch 2000 --split_iter 10 --split_percent 0.05 --split_interval 5
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 2 --batch 2000 --split_iter 10 --split_percent 0.05 --split_interval 5
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 3 --batch 2000 --split_iter 10 --split_percent 0.05 --split_interval 5
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 1 --batch 2000 --split_iter 10 --split_percent 0.0 --split_interval 5 --adapt_split 1
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 2 --batch 2000 --split_iter 10 --split_percent 0.0 --split_interval 5 --adapt_split 1
mpirun -np 4 python -m baselines.split_net.run_exp --expname _2taskfric011_ --seed 3 --batch 2000 --split_iter 10 --split_percent 0.0 --split_interval 5 --adapt_split 1
#fi
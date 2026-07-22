python training.py --dataname roman_empire --lr 0.0001 --dprate 0.0 --dropout 0.0 --wd 1e-5 --wd1 1e-4 
python training.py --dataname amazon_ratings --lr 0.005 --dprate 0.3 --dropout 0.4
python training.py --dataname minesweeper --lr 0.0001 --dprate 0.4 --dropout 0.1
python training.py --is_bns True --act_fn prelu --dataname tolokers --dprate 0.5 --dropout 0.4 --lr 0.01 --wd 1e-5 --wd1 1e-4
python training.py --dataname questions --lr 0.001 --dprate 0.1 --dropout 0.2 --epochs 1000 --is_bns True --wd 1e-5 --wd1 1e-4 --lr1 0.005 
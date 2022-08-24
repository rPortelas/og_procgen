#### Install 

```bash
git clone -b dev/loca_procgen_ppo https://github.com/rPortelas/og_procgen.git
cd og_procgen/
conda env update --name locaprocgen --file train-procgen/environment.yml
conda activate locaprocgen
cd procgen/
pip install -e .
cd ../baselines/
pip install -e .
cd ../train_procgen
pip install -e .
conda install qt==5.9.7
```

#### fix potential protobuf issues with tensorflow
```bash
pip install protobuf==3.19.4
```
#### Launch dummy PPO run on coinrun loca to verify implementation
```bash
 python3 -m train_procgen.train --env_name locacoinrun --distribution_mode easy --result_dir results --exp_name bobo --num_envs 3 --nb_test_episodes 3 --phase_1_len 0.001 --phase_2_len 0.002 --phase_3_len 0.002
```
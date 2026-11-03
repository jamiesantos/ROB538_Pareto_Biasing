import random
import torch
import numpy
import sys
import os
import datetime
import shutil

import CCEA
import td3

if __name__ == '__main__':
    assert len(sys.argv) == 9, "Correct usage: python main.py ccea|td3 rover data_dirpath alg_config env_config seed label traj_write_freq"

    alg_name = sys.argv[1]
    assert alg_name in ['ccea', 'td3'], "Unrecognised alg_name"
    domain_name = sys.argv[2]
    assert domain_name == 'rover', 'Unrecognised domain_name'
    data_dir = sys.argv[3]
    data_dir = data_dir+'/' if data_dir[-1]!='/' else data_dir
    os.makedirs(data_dir, exist_ok=True)
    src_alg_config_filename = sys.argv[4]
    src_env_config_filename = sys.argv[5]
    seed_val = int(sys.argv[6])
    seed_val_str = str(seed_val)
    label = sys.argv[7]
    traj_write_freq = int(sys.argv[8])

    datetime_now_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    data_filename = data_dir+alg_name+'_'+domain_name+'_'+seed_val_str+'_'+label+'_'+datetime_now_string+'_savedata.csv'
    dest_alg_config_filename = data_dir+alg_name+'_'+domain_name+'_'+seed_val_str+'_'+label+'_'+datetime_now_string+'_algconfig.yaml'
    shutil.copyfile(src_alg_config_filename, dest_alg_config_filename)
    dest_env_config_filename = data_dir+alg_name+'_'+domain_name+'_'+seed_val_str+'_'+label+'_'+datetime_now_string+'_envconfig.yaml'
    shutil.copyfile(src_env_config_filename, dest_env_config_filename)

    random.seed(seed_val)
    torch.manual_seed(seed_val)
    numpy.random.seed(seed_val)

    if alg_name == 'ccea':
        alg = CCEA.CCEA(alg_config_filename=dest_alg_config_filename,
                        domain_name=domain_name,
                        data_filename=data_filename,
                        rover_config_filename=dest_env_config_filename)

        for gen in range(alg.num_gens):
            alg.evolve(gen=gen, traj_write_freq=traj_write_freq)
    else:
        alg = td3.TD3Trainer(alg_config_filename=dest_alg_config_filename,
                             rover_config_filename=dest_env_config_filename,
                             data_filename=data_filename)

        for episode in range(alg.num_episodes):
            alg.train_episode(episode=episode, traj_write_freq=traj_write_freq)

        # Save the trained actors and the shared critic next to the run's data
        alg.td3.save(data_dir+alg_name+'_'+domain_name+'_'+seed_val_str+'_'+label+'_'+datetime_now_string+'_model')

import numpy as np
import os
import signal
import subprocess
import datetime
import time


def create_random_search_folder(base_xp_name):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"experiences/random_search_{base_xp_name}_{now}"
    os.mkdir(folder_name)
    return folder_name


def parse_config(exp_name):
    exp_config = {}
    f = open(f'experiences/{exp_name}/exp_config.txt', 'r')
    lines = f.readlines()
    f.close()
    for i in lines:
        values = i[:-1].split("    ")
        exp_config[values[0]] = values[1]
    return exp_config


def generate_new_config(base_config, folder, xp_id):
    new_random_config = {}
    for key, value in base_config.items():
        if key in ['batch_size', 'dp_keep_prob', 'emb_size', 'hidden_size', 'initial_lr', 'num_layers', 'seq_len']:
            new_random_value = np.random.randn() * float(value) / 2
            if key in ['batch_size', 'emb_size', 'hidden_size', 'num_layers', 'seq_len']:
                new_random_value = int(new_random_value)
            new_random_config[key] = new_random_value
        elif key in ['model', 'optimizer']:
            new_random_config[key] = value
    new_random_config['save_dir'] = f"{folder}/{xp_id}_"
    return new_random_config


def start_process_with_config(config):
    command_string = "python ptb-lm.py"
    for key, value in config.items():
        command_string += f" --{key}={value}"
    return subprocess.Popen(command_string)


def monitor_process(process, folder, xp_id, base_ppls):
    need_to_kill = False
    xp_folder = ""
    for xp in os.listdir(folder):
        if xp.startswith(xp_id + "_"):
            xp_folder = xp
    path = f'{folder}/{xp_folder}'
    while True:
        time.sleep(30)
        ppls = parse_log(path)
        current_epoch = len(ppls) - 1
        if current_epoch >= 1:
            if ppls[current_epoch][0] < base_ppls[current_epoch][0] and ppls[current_epoch][1] < base_ppls[current_epoch][1]:
                need_to_kill = True
                break
        if current_epoch == 39:
            break
    if need_to_kill:
        kill_process(process)


def parse_log(path):
    f = open(f'{path}/log.txt', 'r')
    lines = f.readlines()
    f.close()
    ppls = []
    for line in lines:
        values = line.split("\t")
        epoch = int(values[0].split("epoch: ")[1])
        train_ppl = float(values[1].split("train ppl: ")[1])
        val_ppl = float(values[2].split("val ppl: ")[1])
        ppls.append((train_ppl, val_ppl))
    return ppls


def kill_process(process):
    os.killpg(process.pid, signal.SIGTERM)


# ========== MAIN ==========
base_xp_name = "RNN_ADAM"
folder = create_random_search_folder(base_xp_name)
base_config = parse_config(base_xp_name)
base_ppls = parse_log(f"experiences/{base_xp_name}")
xp_id = 0
while True:
    xp_id += 1
    new_config = generate_new_config(base_config, folder, xp_id)
    process = start_process_with_config(new_config)
    monitor_process(process, folder, xp_id, base_ppls)

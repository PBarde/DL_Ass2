import numpy as np
import os
import subprocess
import datetime
import time
import signal


def generate_random_search_experience_name(base_xp_name):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"random_search_{base_xp_name}_{now}"
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


def generate_new_config(base_config, random_search_experience_name, xp_id):
    new_random_config = {}
    for key, value in base_config.items():
        if key in ['batch_size', 'dp_keep_prob', 'emb_size', 'hidden_size', 'initial_lr', 'num_layers', 'seq_len']:
            new_random_value = float(value) + np.random.randn() * float(value) / 2
            if key in ['batch_size', 'emb_size', 'hidden_size', 'num_layers', 'seq_len']:
                new_random_value = max(1, int(new_random_value))
            if key in ['dk_keep_prob']:
                new_random_value = max(0.05, min(0.95, float(new_random_value)))
            new_random_config[key] = new_random_value
        elif key in ['model', 'optimizer']:
            new_random_config[key] = value
    new_random_config['save_dir'] = f"{random_search_experience_name}_{xp_id}_"
    print("generated config:", new_random_config)
    return new_random_config


def start_process_with_config(config):
    command_string = "python ptb-lm.py"
    for key, value in config.items():
        command_string += f" --{key}={value}"
    process = subprocess.Popen(command_string, shell=False)#, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return process


def monitor_process(process, random_search_experience_name, xp_id, base_ppls):
    time.sleep(30)
    kill_process(process)
    return
    need_to_kill = False
    xp_folder = ""
    search_name = f"{random_search_experience_name}_{xp_id}_"
    for xp in os.listdir("./"):
        if xp.startswith(search_name):
            xp_folder = xp
            break
    if xp_folder == "":
        print("Failed to find folder that starts with:", search_name)
        print("in", os.listdir("./"))
    while True:
        ppls = parse_log(xp_folder)
        current_epoch = len(ppls) - 1
        if current_epoch >= 0:
            if ppls[current_epoch][0] > base_ppls[current_epoch][0] and ppls[current_epoch][1] > base_ppls[current_epoch][1]:
                print(f"Stopping training because current ppl values did not beat the ones of the base xp "
                      f"(train: {base_ppls[current_epoch][0]}, val: {base_ppls[current_epoch][1]})")
                need_to_kill = True
                break
        if current_epoch == 39:
            break
        time.sleep(30)
    if need_to_kill:
        kill_process(process)


def parse_log(xp_folder):
    if not os.path.isfile(f'{xp_folder}/log.txt'):
        return []

    f = open(f'{xp_folder}/log.txt', 'r')
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
    os.getpgid()
    os.killpg(os.getpgid(process.pid), signal.SIGINT)
    # try:
    #     process.kill()
    # except OSError:
    #     print("Failed to kill the process")


# ========== MAIN ==========
base_xp_name = "RNN_ADAM"
random_search_experience_name = generate_random_search_experience_name(base_xp_name)
base_config = parse_config(base_xp_name)
base_ppls = parse_log(f"experiences/{base_xp_name}")
xp_id = 0
while True:
    xp_id += 1
    print(f"Generating experience {xp_id}")
    new_config = generate_new_config(base_config, random_search_experience_name, xp_id)
    process = start_process_with_config(new_config)
    monitor_process(process, random_search_experience_name, xp_id, base_ppls)

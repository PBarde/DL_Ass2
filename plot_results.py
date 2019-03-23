import numpy as np
import matplotlib.pyplot as plt
import os

FIRST = True


def parse_config():
    exp_config = {}
    f = open(f'experiences/{EXP_NAME}/exp_config.txt', 'r')
    lines = f.readlines()
    f.close()
    for i in lines:
        values = i[:-1].split("    ")
        exp_config[values[0]] = values[1]
    return exp_config


def parse_log():
    wall_clock_time = 0
    wall_clock_times = []
    best_ppl_pair = ()
    cumulative_wall_clock_time = 0
    f = open(f'experiences/{EXP_NAME}/log.txt', 'r')
    lines = f.readlines()
    f.close()
    for i in lines:
        values = i.split("\t")
        train_ppl = float(values[1].split("train ppl: ")[1])
        val_ppl = float(values[2].split("val ppl: ")[1])
        if best_ppl_pair == () or val_ppl < best_ppl_pair[1]:
            best_ppl_pair = (train_ppl, val_ppl)
        wall_clock_time = float(values[4][:-1].split("time (s) spent in epoch: ")[1]) / 3600
        cumulative_wall_clock_time += wall_clock_time
        wall_clock_times.append(cumulative_wall_clock_time)
    if len(wall_clock_times) < 40:
        for i in range(40 - len(wall_clock_times)):
            wall_clock_times.append(wall_clock_times[-1] + wall_clock_time)
    return best_ppl_pair, wall_clock_times


def print_experience(EXP_NAME, exp_config, ppls, wall_clock_times):
    headers = "EXP_NAME\t"
    config_str = ""
    for key, value in exp_config.items():
        if key in ["code_file", "data", "debug", "evaluate", "save_best", "save_dir", "seed"]:
            continue
        headers += f"{key}\t"
        config_str += f"{value}\t"
    if FIRST:
        print(f"{headers}train_ppl\tval_ppl")
    print(f'{EXP_NAME}\t{config_str}{ppls[0]}\t{ppls[1]}')


def parse_learning_curves(exp_config):
    epochs = int(exp_config["num_epochs"])
    curves = {}
    if os.path.exists(f'experiences/{EXP_NAME}/learning_curves.npy'):
        learning_curves = np.load(f'experiences/{EXP_NAME}/learning_curves.npy')[()]
        for curve in learning_curves:
            array = np.array(learning_curves[curve])

            if "losses" in curve:
                steps = int(array.shape[0] / epochs)
                array = array[steps-1::steps]

            curves[curve] = array
    else:
        f = open(f'experiences/{EXP_NAME}/log.txt', 'r')
        lines = f.readlines()
        f.close()
        train_ppls = []
        val_ppls = []
        for line in lines:
            values = line.split("\t")
            epoch = int(values[0].split("epoch: ")[1])
            train_ppls.append(float(values[1].split("train ppl: ")[1]))
            val_ppls.append(float(values[2].split("val ppl: ")[1]))
        if len(train_ppls) < 40:
            last_train_ppl = train_ppls[-1]
            last_val_ppl = val_ppls[-1]
            for i in range(40 - len(train_ppls)):
                train_ppls.append(last_train_ppl)
                val_ppls.append(last_val_ppl)

        curves["train_ppls"] = np.array(train_ppls)
        curves["val_ppls"] = np.array(val_ppls)
        curves["train_losses"] = np.zeros(len(train_ppls))
        curves["val_losses"] = np.zeros(len(val_ppls))
    return curves


def plot_curves(curves, wall_clock_times):
    # train & val ppl per epoch
    fig, ax = plt.subplots()
    ax.plot(curves["train_ppls"], label="train_ppls")
    ax.legend()
    ax.plot(curves["val_ppls"], label="val_ppls")
    ax.legend()
    ax.set(xlabel='epoch', ylabel='ppl', title=EXP_NAME)
    ax.grid()
    plt.savefig(f'experiences/{EXP_NAME}/{EXP_NAME.lower()}_epoch.png')
    # plt.show()
    plt.close()

    # train & val ppl by wall-clock-time
    fig, ax = plt.subplots()
    ax.plot(wall_clock_times, curves["train_ppls"], label="train_ppls")
    ax.legend()
    ax.plot(wall_clock_times, curves["val_ppls"], label="val_ppls")
    ax.legend()
    ax.set(xlabel='wall-clock-time (h)', ylabel='ppl', title=EXP_NAME)
    ax.grid()
    plt.savefig(f'experiences/{EXP_NAME}/{EXP_NAME.lower()}_wall_clock_time.png')
    # plt.show()
    plt.close()

    # train & val loss per epoch
    fig, ax = plt.subplots()
    ax.plot(curves["train_losses"], label="train_losses")
    ax.legend()
    ax.plot(curves["val_losses"], label="val_losses")
    ax.legend()
    ax.set(xlabel='epoch', ylabel='loss', title=EXP_NAME)
    ax.grid()
    # plt.show()
    plt.close()

    # train & val loss by wall-clock-time
    fig, ax = plt.subplots()
    ax.plot(wall_clock_times, curves["train_losses"], label="train_losses")
    ax.legend()
    ax.plot(wall_clock_times, curves["val_losses"], label="val_losses")
    ax.legend()
    ax.set(xlabel='wall-clock-time (h)', ylabel='loss', title=EXP_NAME)
    ax.grid()
    # plt.show()
    plt.close()


for EXP_NAME in os.listdir('experiences'):
    exp_config = parse_config()
    ppls, wall_clock_times = parse_log()
    print_experience(EXP_NAME, exp_config, ppls, wall_clock_times)
    curves = parse_learning_curves(exp_config)
    plot_curves(curves, wall_clock_times)
    FIRST = False

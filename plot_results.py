import numpy as np
import matplotlib.pyplot as plt
import os


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
        wall_clock_time = float(values[4][:-1].split("time (s) spent in epoch: ")[1])
        cumulative_wall_clock_time += wall_clock_time / 3600
        wall_clock_times.append(cumulative_wall_clock_time)
    print(f'{EXP_NAME}\t{best_ppl_pair[0]}\t{best_ppl_pair[1]}')
    return best_ppl_pair, wall_clock_times


def parse_learning_curves(exp_config):
    epochs = int(exp_config["num_epochs"])
    learning_curves = np.load(f'experiences/{EXP_NAME}/learning_curves.npy')[()]
    curves = {}
    for curve in learning_curves:
        array = np.array(learning_curves[curve])

        if "losses" in curve:
            steps = int(array.shape[0] / epochs)
            array = array[steps-1::steps]

        curves[curve] = array
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
    curves = parse_learning_curves(exp_config)
    plot_curves(curves, wall_clock_times)

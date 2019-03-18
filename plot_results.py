import numpy as np
import matplotlib.pyplot as plt

# EXP_NAME = "RNN_base"
EXP_NAME = "RNN_SGD"


def parse_config():
    exp_config = {}
    f = open(f'experiences/{EXP_NAME}/exp_config.txt', 'r')
    lines = f.readlines()
    f.close()
    for i in lines:
        values = i[:-1].split("    ")
        exp_config[values[0]] = values[1]
    print(exp_config)
    return exp_config


def parse_log():
    wall_clock_times = []
    cumulative_wall_clock_time = 0
    f = open(f'experiences/{EXP_NAME}/log.txt', 'r')
    lines = f.readlines()
    f.close()
    for i in lines:
        values = i[:-1].split("time (s) spent in epoch: ")
        cumulative_wall_clock_time += float(values[1]) / 3600
        wall_clock_times.append(cumulative_wall_clock_time)
    print(wall_clock_times)
    return wall_clock_times


def parse_learning_curves(exp_config):
    epochs = int(exp_config["num_epochs"])
    learning_curves = np.load(f'experiences/{EXP_NAME}/learning_curves.npy')[()]
    curves = {}
    for curve in learning_curves:
        print(curve)
        array = np.array(learning_curves[curve])
        print(array.shape)

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
    ax.set(xlabel='epoch', ylabel='ppl', title='ppl per epoch')
    ax.grid()
    plt.show()

    # train & val ppl by wall-clock-time
    fig, ax = plt.subplots()
    ax.plot(wall_clock_times, curves["train_ppls"], label="train_ppls")
    ax.legend()
    ax.plot(wall_clock_times, curves["val_ppls"], label="val_ppls")
    ax.legend()
    ax.set(xlabel='wall-clock-time (h)', ylabel='ppl', title='ppl by wall-clock-time')
    ax.grid()
    plt.show()

    # train & val loss per epoch
    fig, ax = plt.subplots()
    ax.plot(curves["train_losses"], label="train_losses")
    ax.legend()
    ax.plot(curves["val_losses"], label="val_losses")
    ax.legend()
    ax.set(xlabel='epoch', ylabel='loss', title='loss per epoch')
    ax.grid()
    plt.show()

    # train & val loss by wall-clock-time
    fig, ax = plt.subplots()
    ax.plot(wall_clock_times, curves["train_losses"], label="train_losses")
    ax.legend()
    ax.plot(wall_clock_times, curves["val_losses"], label="val_losses")
    ax.legend()
    ax.set(xlabel='wall-clock-time (h)', ylabel='loss', title='loss by wall-clock-time')
    ax.grid()
    plt.show()


exp_config = parse_config()
wall_clock_times = parse_log()
curves = parse_learning_curves(exp_config)
plot_curves(curves, wall_clock_times)

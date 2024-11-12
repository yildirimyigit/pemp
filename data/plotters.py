import matplotlib.pyplot as plt


# plot train trajectories on the left and validation trajectories on the right
def plot_train_val(num_demos, x_train, y_train, num_val, x_val, y_val):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(num_demos):
        axs[0].plot(x_train[i, :, 0], y_train[i, :, 0], label=f"Demo {i}")
    for i in range(num_val):
        axs[0].plot(x_val[i, :, 0], y_val[i, :, 0], label=f"Val {i}", color='black', linestyle='dashed')
        axs[1].plot(x_val[i, :, 0], y_val[i, :, 0], label=f"Val {i}", color='black', linestyle='dashed')
    axs[0].set_title("Training")
    axs[1].set_title("Validation")
    axs[0].grid(True)
    axs[1].grid(True)

    plt.show()


def plot_train_test(num_demos, x_train, y_train, num_test, x_test, y_test):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(num_demos):
        axs[0].plot(x_train[i, :, 0], y_train[i, :, 0], label=f"Demo {i}")
    for i in range(num_test):
        axs[0].plot(x_test[i, :, 0], y_test[i, :, 0], label=f"Test {i}", color='black', linestyle='dashed')
        axs[1].plot(x_test[i, :, 0], y_test[i, :, 0], label=f"Test {i}", color='black', linestyle='dashed')
    axs[0].set_title("Training")
    axs[1].set_title("Test")
    axs[0].grid(True)
    axs[1].grid(True)

    plt.show()

def plot_trajs(x, y):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i in range(y.shape[0]):
        ax.plot(x[i, :, 0], y[i, :, 0], color='#4d4d4d', alpha=0.25)
    ax.grid(True)
    plt.show()
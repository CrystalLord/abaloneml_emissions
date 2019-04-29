import matplotlib.pyplot as plt

def plot_nestedcv(full_train_mse, test_mse, alphas, title="", regr_name="regr"):

    plt.plot(full_train_mse * 100, label='Full Train MSE')
    plt.plot(test_mse * 100, label='Test MSE')
    plt.title(title)
    plt.ylabel("Mean Squared Errror (x100)")
    plt.xlabel('Fold Index')
    plt.legend()
    plt.ylim((0,0.04))
    plt.savefig(f"{regr_name}_cv_plot.png", dpi=196)
    #plt.show()

    # Show alphas.
    plt.plot(alphas, '--', label='Alpha')
    #plt.show()

def plot_pred_v_reality(real, pred, base=None, title="", regr_name="regr"):
    """Plots a scatter plot with matching residual lines for both predictions
    and the true data.

    Args:
        times (np.ndarray):
        real (np.ndarray):
        pred (np.ndarray):
    """
    times = list(range(len(real)))
    plt.vlines(times, real, pred, linestyle='dashed', alpha=0.5, zorder=-2)
    plt.scatter(times, real, s=30, marker='.', label="True")
    plt.scatter(times, pred, s=30, marker='x', label="Prediction")
    if base is not None:
        plt.plot(times, base, label='Baseline', zorder=-1, color='k')
    plt.title(title)
    plt.xlabel("Test Sample Index")
    plt.ylabel('Peak Ozone, Parts Per Million')
    #plt.grid()
    plt.legend()
    plt.savefig(f"{regr_name}_pred_true.png", dpi=196)
    plt.show()


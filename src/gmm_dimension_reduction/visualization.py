import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.mixture import GaussianMixture

from gmm_dimension_reduction import GaussianMixtureModel
from gmm_dimension_reduction.slicing import slice_gmm


def plot_gmm(gmm, ax=None, datapoints=None, extends=None):
    def __get_range(min_value, max_value, margin=0.1):
        return (max_value - min_value) * margin

    if datapoints is not None:
        datapoints = [datapoints.iloc[:, 0], datapoints.iloc[:, 1]]
    if not extends:
        if datapoints is not None:
            extends = [
                (point.min() - __get_range(point.min(), point.max()),
                 point.max() + __get_range(point.min(), point.max()))
                for point in datapoints]
        else:
            extends = [(-4., 4.), (-1.5, 1.5)]
    x = np.linspace(*extends[0])
    y = np.linspace(*extends[1])
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = np.array(gmm.predict(XX, "selfmade"))
    Z = Z.reshape(X.shape)
    if ax is not None:
        if datapoints is not None:
            ax.scatter(datapoints[0], datapoints[1])
        ax.contour(X, Y, Z)
    else:
        if datapoints is not None:
            plt.scatter(datapoints[0], datapoints[1])
        plt.contour(X, Y, Z)


def slicing_plot_matrix(data, **kwargs):
    def __create_gmm_plot(ax, gm, df, features, n_components=2):
        weights_, means_, covariances_ = slice_gmm(features, gm.weights_, gm.means_, gm.covariances_)
        gmm = GaussianMixtureModel.GaussianMixtureModel(weights_, means_, covariances_)
        plot_gmm(gmm, ax=ax, datapoints=df[df.columns[[i - 1 for i in features]]])

    # https://stackoverflow.com/questions/7941207/is-there-a-function-to-make-scatterplot-matrices-in-matplotlib
    _, numvars = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    gm = GaussianMixture(n_components=3, random_state=0).fit(data)

    # Plot the data.
    for i, j in tqdm(zip(*np.triu_indices_from(axes, k=1))):
        for x, y in [(i, j), (j, i)]:
            __create_gmm_plot(axes[x, y], gm, data, [x, y])

    names = data.columns
    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)
    return fig

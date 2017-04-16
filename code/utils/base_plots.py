import pylab as pb
import numpy as np


def ax_default(fignum, ax):
    if ax is None:
        fig = pb.figure(fignum)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    return (fig, ax)


def meanplot(
    x,
    mu,
    color,
    ax=None,
    fignum=None,
    linewidth=4,
    linestyle='solid',
    **kw
    ):
    (_, axes) = ax_default(fignum, ax)
    return axes.plot(
        x,
        mu,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        **kw
        )


def gpplot(
    x,
    mu,
    lower,
    upper,
    edgecol,
    fillcol,
    ax=None,
    fignum=None,
    linestyle='solid',
    **kwargs
    ):
    (_, axes) = ax_default(fignum, ax)

    mu = mu.flatten()
    x = x.flatten()
    lower = lower.flatten()
    upper = upper.flatten()

    plots = []


    plots.append(meanplot(x, mu, edgecol, axes, linestyle=linestyle))


    kwargs['linewidth'] = 0.5
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 0.3
    plots.append(axes.fill(np.hstack((x, x[::-1])), np.hstack((upper,
                 lower[::-1])), color=fillcol, linestyle=linestyle,
                 **kwargs))


    plots.append(meanplot(
        x,
        upper,
        color=edgecol,
        linewidth=0.4,
        ax=axes,
        linestyle=linestyle,
        ))
    plots.append(meanplot(
        x,
        lower,
        color=edgecol,
        linewidth=0.4,
        ax=axes,
        linestyle=linestyle,
        ))

    return plots


def removeRightTicks(ax=None):
    ax = ax or pb.gca()
    for (i, line) in enumerate(ax.get_yticklines()):
        if i % 2 == 1:  # odd indices
            line.set_visible(False)


def removeUpperTicks(ax=None):
    ax = ax or pb.gca()
    for (i, line) in enumerate(ax.get_xticklines()):
        if i % 2 == 1:  # odd indices
            line.set_visible(False)


def fewerXticks(ax=None, divideby=2):
    ax = ax or pb.gca()
    ax.set_xticks(ax.get_xticks()[::divideby])


def align_subplots(
    N,
    M,
    xlim=None,
    ylim=None,
    ):

    if xlim is None:
        xlim = [np.inf, -np.inf]
        for i in range(N * M):
            pb.subplot(N, M, i + 1)
            xlim[0] = min(xlim[0], pb.xlim()[0])
            xlim[1] = max(xlim[1], pb.xlim()[1])
    if ylim is None:
        ylim = [np.inf, -np.inf]
        for i in range(N * M):
            pb.subplot(N, M, i + 1)
            ylim[0] = min(ylim[0], pb.ylim()[0])
            ylim[1] = max(ylim[1], pb.ylim()[1])

    for i in range(N * M):
        pb.subplot(N, M, i + 1)
        pb.xlim(xlim)
        pb.ylim(ylim)
        if i % M:
            pb.yticks([])
        else:
            removeRightTicks()
        if i < M * (N - 1):
            pb.xticks([])
        else:
            removeUpperTicks()


def align_subplot_array(axes, xlim=None, ylim=None):

    if xlim is None:
        xlim = [np.inf, -np.inf]
        for ax in axes.flatten():
            xlim[0] = min(xlim[0], ax.get_xlim()[0])
            xlim[1] = max(xlim[1], ax.get_xlim()[1])
    if ylim is None:
        ylim = [np.inf, -np.inf]
        for ax in axes.flatten():
            ylim[0] = min(ylim[0], ax.get_ylim()[0])
            ylim[1] = max(ylim[1], ax.get_ylim()[1])

    (N, M) = axes.shape
    for (i, ax) in enumerate(axes.flatten()):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if i % M:
            ax.set_yticks([])
        else:
            removeRightTicks(ax)
        if i < M * (N - 1):
            ax.set_xticks([])
        else:
            removeUpperTicks(ax)


def x_frame1D(X, plot_limits=None, resolution=None):

    assert X.shape[1] == 1, \
        'x_frame1D is defined for one-dimensional inputs'
    if plot_limits is None:
        (xmin, xmax) = (X.min(0), X.max(0))
        (xmin, xmax) = (xmin - 0.2 * (xmax - xmin), xmax + 0.2 * (xmax
                        - xmin))
    elif len(plot_limits) == 2:
        (xmin, xmax) = plot_limits
    else:
        raise ValueError, 'Bad limits for plotting'

    Xnew = np.linspace(xmin, xmax, resolution or 200)[:, None]
    return (Xnew, xmin, xmax)


def x_frame2D(X, plot_limits=None, resolution=None):

    assert X.shape[1] == 2, \
        'x_frame2D is defined for two-dimensional inputs'
    if plot_limits is None:
        (xmin, xmax) = (X.min(0), X.max(0))
        (xmin, xmax) = (xmin - 0.2 * (xmax - xmin), xmax + 0.2 * (xmax
                        - xmin))
    elif len(plot_limits) == 2:
        (xmin, xmax) = plot_limits
    else:
        raise ValueError, 'Bad limits for plotting'

    resolution = resolution or 50
    (xx, yy) = np.mgrid[xmin[0]:xmax[0]:1j * resolution, xmin[1]:
                        xmax[1]:1j * resolution]
    Xnew = np.vstack((xx.flatten(), yy.flatten())).T
    return (Xnew, xx, yy, xmin, xmax)



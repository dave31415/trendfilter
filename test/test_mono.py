import numpy as np
from trendfilter import trend_filter
from bokeh.plotting import figure, show
from scipy.interpolate import interp1d

show_plot = True


def test_mono():
    noise = 0.2
    np.random.seed(420)

    x = np.linspace(0, 10, 80)
    n = len(x)
    y = np.sqrt(x)
    y_noisy = y + noise * np.random.randn(n)
    y_noisy[20] += 3

    y_fit = trend_filter(x, y_noisy, monotonic=True)

    plot = figure()
    plot.circle(x, y_noisy)
    plot.line(x, y_noisy)

    plot.line(x, y_fit, color='red')

    f = interp1d(x, y_fit, fill_value="extrapolate")

    xx = np.linspace(x.min() - 1, x.max() + 2, 100)
    plot.line(xx, f(xx), color='green', line_dash='dashed')

    if show_plot:
        show(plot)

import numpy as np
from trendfilter import trend_filter
from bokeh.plotting import figure, show
from bokeh.io import output_file
from scipy.interpolate import interp1d
from tempfile import NamedTemporaryFile
from numpy import ndarray

show_plot = True


def prep_data():
    noise = 0.2
    np.random.seed(420)

    # make a set of x, y points
    # y = sqrt(x) plus noise

    x = np.linspace(0, 10, 80)
    n = len(x)
    y = np.sqrt(x)
    y_noisy = y + noise * np.random.randn(n)

    # add an outlier points
    y_noisy[20] += 3
    y_noisy[60] += 2

    return x, y_noisy


def plot_model(title, **kwargs):
    file = NamedTemporaryFile().name+'.html'
    output_file(file)
    x, y_noisy = prep_data()

    # fit a monotonic increasing function
    y_fit = trend_filter(x, y_noisy, **kwargs)

    assert isinstance(y_fit, ndarray)

    plot = figure(title=title)
    plot.circle(x, y_noisy)
    plot.line(x, y_noisy)

    plot.line(x, y_fit, color='red')

    # create an interpolation function for the model
    # can also just give it return_function=True
    # and get back the interpolation function instead of the
    # model evaluated at points

    f = interp1d(x, y_fit, fill_value="extrapolate")

    # over-plot the function, showing the extrapolation too
    xx = np.linspace(x.min() - 1, x.max() + 2, 500)
    plot.line(xx, f(xx), color='green', line_dash='dashed')

    if show_plot:
        show(plot)


def test_base():
    title = 'Base model, no regularization'
    plot_model(title)


def test_mono():
    title = 'Best monotonic increasing function'
    plot_model(title, monotonic=True)


def test_l1_trend_filter():
    title = 'L1 Trend Filter Model'
    plot_model(title, l_norm=1, alpha_1=0.2)


def test_l1_trend_filter_mono():
    title = 'L1 Trend Filter Model, Monotonic'
    plot_model(title, l_norm=1, alpha_1=0.2, monotonic=True)


def test_l1_trend_filter_more_reg():
    title = 'L1 Trend Filter Model, More regularization'
    plot_model(title, l_norm=1, alpha_1=2.0)


def test_l1_trend_filter_steps():
    title = 'L1 Trend Filter Model, Stair steps'
    plot_model(title, l_norm=1, alpha_0=8.0)


def test_l2_smooth():
    title = 'L2 Trend Filter Model, Smooth'
    plot_model(title, l_norm=2, alpha_1=1.0)


def test_l1_piecewise_quadratic():
    title = 'L1 Trend Filter Model, Piecewise quadratic, constrain zero'
    plot_model(title, l_norm=1, alpha_2=3.0, constrain_zero=True)

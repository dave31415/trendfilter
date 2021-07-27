import numpy as np
from trendfilter import trend_filter
from bokeh.plotting import figure, show
from bokeh.io import output_file
from scipy.interpolate import interp1d
from tempfile import NamedTemporaryFile
from numpy import ndarray

show_plot = True
tolerance = 1/(10**8)


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

    result = trend_filter(x, y_noisy, **kwargs)
    y_fit = result['y_fit']
    assert isinstance(y_fit, ndarray)

    plot = figure(title=title)
    plot.circle(x, y_noisy)
    plot.line(x, y_noisy)

    plot.line(x, y_fit, color='red')

    f = result['function']

    # over-plot the function, showing the extrapolation too
    xx = np.linspace(x.min() - 1, x.max() + 2, 500)
    plot.line(xx, f(xx), color='green', line_dash='dashed')

    if show_plot:
        show(plot)

    obj = result['objective_total'].value
    print('objective %s, %s' % (obj, title))

    return obj


def test_base():
    title = 'Base model, no regularization'
    obj = plot_model(title)
    assert obj < tolerance


def test_mono():
    title = 'Best monotonic increasing function'
    obj = plot_model(title, monotonic=True)
    assert abs(obj - 10.39020002241298) < tolerance


def test_l1_trend_filter():
    title = 'L1 Trend Filter Model'
    obj = plot_model(title, l_norm=1, alpha_2=0.2)
    assert abs(obj - 12.045109020871877) < tolerance


def test_l1_trend_filter_mono():
    title = 'L1 Trend Filter Model, Monotonic'
    obj = plot_model(title, l_norm=1, alpha_2=0.2, monotonic=True)
    assert abs(obj - 12.052960090588234) < tolerance


def test_l1_trend_filter_more_reg():
    title = 'L1 Trend Filter Model, More regularization'
    obj = plot_model(title, l_norm=1, alpha_2=2.0)
    assert abs(obj - 13.16869494642045) < tolerance


def test_l1_trend_filter_steps():
    title = 'L1 Trend Filter Model, Stair steps, Constrain zero'
    obj = plot_model(title, l_norm=1, alpha_1=1.0, constrain_zero=True)
    assert abs(obj - 33.64932708644826) < tolerance


def test_smooth():
    title = 'L2 Smooth'
    obj = plot_model(title, l_norm=2, alpha_2=2.0)
    assert abs(obj - 11.971096302251315) < tolerance

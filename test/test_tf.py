import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from tempfile import NamedTemporaryFile
from numpy import ndarray
from trendfilter import trend_filter
from trendfilter.get_example_data import get_example_data


show_plot = True
tolerance = 1/(10**8)


def plot_model(x, y_noisy, title, **kwargs):
    file = NamedTemporaryFile().name+'.html'
    output_file(file)

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
    x, y_noisy = get_example_data()
    title = 'Base model, no regularization'
    obj = plot_model(x, y_noisy, title)
    assert obj < tolerance


def test_mono():
    x, y_noisy = get_example_data()
    title = 'Best monotonic increasing function'
    obj = plot_model(x, y_noisy,title, monotonic=True)
    assert abs(obj - 10.39020002241298) < tolerance


def test_l1_trend_filter():
    x, y_noisy = get_example_data()
    title = 'L1 Trend Filter Model'
    obj = plot_model(x, y_noisy, title, l_norm=1, alpha_2=0.2)
    assert abs(obj - 12.045109020871877) < tolerance


def test_l1_trend_filter_mono():
    x, y_noisy = get_example_data()
    title = 'L1 Trend Filter Model, Monotonic'
    obj = plot_model(x, y_noisy, title, l_norm=1, alpha_2=0.2, monotonic=True)
    assert abs(obj - 12.052960090588234) < tolerance


def test_l1_trend_filter_more_reg():
    x, y_noisy = get_example_data()
    title = 'L1 Trend Filter Model, More regularization'
    obj = plot_model(x, y_noisy, title, l_norm=1, alpha_2=2.0)
    assert abs(obj - 13.16869494642045) < tolerance


def test_l1_trend_filter_steps():
    x, y_noisy = get_example_data()
    title = 'L1 Trend Filter Model, Stair steps, Constrain zero'
    obj = plot_model(x, y_noisy, title, l_norm=1, alpha_1=1.0, constrain_zero=True)
    assert abs(obj - 33.64932708644826) < tolerance


def test_smooth():
    x, y_noisy = get_example_data()
    title = 'L2 Smooth'
    obj = plot_model(x, y_noisy, title, l_norm=2, alpha_2=2.0)
    assert abs(obj - 11.971096302251315) < tolerance

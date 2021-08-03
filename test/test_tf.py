import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from tempfile import NamedTemporaryFile
from numpy import ndarray
from trendfilter import trend_filter
from trendfilter.get_example_data import get_example_data
from trendfilter.plot_model import plot_model

show_plot = True
tolerance = 1/(10**8)


def test_base():
    x, y_noisy = get_example_data()
    title = 'Base model, no regularization'
    result = trend_filter(x, y_noisy)
    plot_model(result, title=title)
    obj = result['objective_total'].value
    print('objective %s, %s' % (obj, title))
    assert obj < tolerance


def test_mono():
    x, y_noisy = get_example_data()
    title = 'Best monotonic increasing function'
    result = trend_filter(x, y_noisy, monotonic=True)
    plot_model(result, title=title, show_extrap=True, extrap_max=3)
    obj = result['objective_total'].value
    print('objective %s, %s' % (obj, title))
    assert abs(obj - 10.39020002241298) < tolerance


def test_l1_trend_filter():
    x, y_noisy = get_example_data()
    title = 'L1 Trend Filter Model'
    result = trend_filter(x, y_noisy, l_norm=1, alpha_2=0.2)
    plot_model(result, title=title, show_extrap=True, extrap_max=3)
    obj = result['objective_total'].value
    print('objective %s, %s' % (obj, title))
    assert abs(obj - 12.044960558386068) < tolerance


def test_l1_trend_filter_mono():
    x, y_noisy = get_example_data()
    title = 'L1 Trend Filter Model, Monotonic'
    result = trend_filter(x, y_noisy, l_norm=1, alpha_2=0.2, monotonic=True)
    plot_model(result, title=title, show_extrap=True, extrap_max=3)
    obj = result['objective_total'].value
    print('objective %s, %s' % (obj, title))
    assert abs(obj - 12.052821372357789) < tolerance


def test_l1_trend_filter_more_reg():
    x, y_noisy = get_example_data()
    title = 'L1 Trend Filter Model, More regularization'
    result = trend_filter(x, y_noisy,  l_norm=1, alpha_2=2.0)
    plot_model(result, title=title, show_extrap=True, extrap_max=3)
    obj = result['objective_total'].value
    print('objective %s, %s' % (obj, title))
    assert abs(obj - 13.167173504429053) < tolerance


def test_l1_trend_filter_steps():
    x, y_noisy = get_example_data()
    title = 'L1 Trend Filter Model, Stair steps, Constrain zero'
    result = trend_filter(x, y_noisy, l_norm=1, alpha_1=1.0, constrain_zero=True)
    plot_model(result, title=title, show_extrap=True, extrap_max=3)
    obj = result['objective_total'].value
    print('objective %s, %s' % (obj, title))
    assert abs(obj - 33.649325762864926) < tolerance


def test_smooth():
    x, y_noisy = get_example_data()
    title = 'L2 Smooth'
    result = trend_filter(x, y_noisy, l_norm=2, alpha_2=2.0)
    plot_model(result, title=title, show_extrap=True, extrap_max=3)
    obj = result['objective_total'].value
    print('objective %s, %s' % (obj, title))
    assert abs(obj - 11.971096302251315) < tolerance

import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from tempfile import NamedTemporaryFile
from numpy import ndarray
from trendfilter import trend_filter
from trendfilter.get_example_data import get_example_data_seasonal, deviation_mapping
from trendfilter.plot_model import plot_model

show_plot = True
tolerance = 1e-8


def test_l1_trend_filter():
    x, y_noisy = get_example_data_seasonal()
    title = 'L1 Trend Filter Model. No seasonality in model.'

    result = trend_filter(x, y_noisy,  l_norm=1, alpha_2=4.0)
    plot_model(result, title=title, show_extrap=True, extrap_max=40)
    obj = result['objective_total'].value
    print('objective', obj, title)
    assert abs(obj-44.42037415318625) < tolerance


def test_with_seasonality():
    x, y_noisy = get_example_data_seasonal()
    title = 'L1 Trend Filter Model. With seasonality in model.'
    mapping = deviation_mapping

    linear_deviation = {'mapping': mapping,
                        'name': 'seasonal_term',
                        'n_vars': 12,
                        'alpha': 0.1}

    linear_deviations = [linear_deviation]

    result = trend_filter(x, y_noisy, l_norm=1, alpha_2=4.0, linear_deviations=linear_deviations)
    plot_model(result, title=title, show_extrap=True, extrap_max=40, show_base=True)
    obj = result['objective_total'].value
    print('objective', obj, title)
    assert abs(obj - 28.311789860224124) < tolerance

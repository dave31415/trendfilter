import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from tempfile import NamedTemporaryFile
from numpy import ndarray
from trendfilter import trend_filter
from trendfilter.get_example_data import get_example_data_seasonal, deviation_mapping

show_plot = True


def plot_model(x, y_noisy, title, show_extras=False, **kwargs):
    file = NamedTemporaryFile().name+'.html'
    output_file(file)

    result = trend_filter(x, y_noisy, **kwargs)
    y_fit = result['y_fit']

    assert isinstance(y_fit, ndarray)

    plot = figure(title=title, width=900, height=600, y_range=(0, 12))
    plot.circle(x, y_noisy, legend='data')
    plot.line(x, y_noisy)

    plot.line(x, y_fit, color='red', legend='model')

    # over-plot the function, showing the extrapolation too
    f = result['function']
    x_min = x.max() + 1
    x_max = x.max() + 40
    xx = np.arange(x_min, x_max)
    plot.line(xx, f(xx), color='green', legend='model extrapolation')

    if show_extras:
        f_base = result['function_base']
        xxx = np.arange(x.min(), x_max)
        plot.line(xxx, f_base(xxx), color='black', legend='base model')

    if show_plot:
        show(plot)


def test_l1_trend_filter():
    x, y_noisy = get_example_data_seasonal()
    title = 'L1 Trend Filter Model. No seasonality in model.'
    plot_model(x, y_noisy, title, l_norm=1, alpha_2=4.0)


def test_with_seasonality():
    x, y_noisy = get_example_data_seasonal()
    title = 'L1 Trend Filter Model. No seasonality in model.'
    mapping = deviation_mapping

    linear_deviation = {'mapping': mapping,
                        'name': 'seasonal_term',
                        'n_vars': 12,
                        'alpha': 0.0}

    linear_deviations = [linear_deviation]

    plot_model(x, y_noisy, title, show_extras=True, l_norm=1, alpha_2=4.0, linear_deviations=linear_deviations)

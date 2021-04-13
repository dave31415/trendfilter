import numpy as np
import cvxpy
from bokeh.plotting import figure, show
from scipy.interpolate import interp1d
from trendfilter.derivatives import second_derivative_matrix_nes, first_derv_nes_cvxpy


def cumulative_matrix(n):
    """
    Matrix of nxn dimension that when multiplied with a vector
        results in the cumulative sum
    :param n: dimension
    :return: cumulative sum matrix
    """
    return np.matrix(np.tril(np.ones((n, n))))


def trend_filter(x, y, y_err=None, alpha_2=0.0,
                 alpha_1=0.0, alpha_0=0.0, l_norm=2,
                 constrain_zero=False, monotonic=False,
                 return_function=False):
    """
    :param x: The x-value, numpy array
    :param y: The y variable, numpy array
    :param y_err: The y_err variable, numpy array
        Default to 1
    :param alpha_0: Regularization against non-zero slope (first derivative)
        Setting this very high will create a series of steps (if L1)
    :param alpha_1: Regularization against non-constant slope (second derivative)
        Setting this very high will result in piecewise linear model (if L1)
    :param alpha_2: Regularization against (third derivative)
     :param l_norm: 1 or 2 to use either L1 or L2 norm
    :param constrain_zero: If True constrains the model to be zero at origin
        Default False
    :param monotonic: If set to True, will result in a monotonically
        increasing function. Default is False.
    :param return_function: If True returns an interpolating function rather
        than the fit value, Default False
    :return: The fit model array
    """

    assert l_norm in [1, 2]
    n = len(x)

    # get the y_err is not supplied
    assert len(y) == n
    if y_err is None:
        y_err = np.ones(n)
    else:
        assert len(y_err) == n

    # the objective function
    obj_func, model = get_obj_func_model(y, y_err=y_err, monotonic=monotonic)

    y_var = obj_func.variables()[0]

    # the regularization
    reg = get_reg(x, y_var, l_norm, alpha_0, alpha_1, alpha_2)

    # the total objective function with regularization
    obj_total = obj_func + reg

    # The objective
    obj = cvxpy.Minimize(obj_total)

    # Get the constraints if any
    constraints = []
    if constrain_zero:
        constraints.append(y_var[0] == 0)

    # define and solve the problem
    problem = cvxpy.Problem(obj, constraints=constraints)
    problem.solve()

    # return the model fit values
    y_fit = model.value

    if return_function:
        # return the interpolating function
        return interp1d(x, y_fit, fill_value="extrapolate")
    else:
        # return the actual model values
        return y_fit


def get_reg(x, y_var, l_norm, alpha_0, alpha_1, alpha_2):
    """
    Get the regularization term
    :param x: The x-value, numpy array
    :param y_var: The y variable, cvxpy.Variable(n)
    :param l_norm: 1 or 2 to use either L1 or L2 norm
    :param alpha_0: Regularization against non-zero slope (first derivative)
        Setting this very high will create a series of steps (if L1)
    :param alpha_1: Regularization against non-constant slope (second derivative)
        Setting this very high will result in piecewise linear model (if L1)
    :param alpha_2: Regularization against (third derivative)
        Setting this very high will result in piecewise quadratic (if L1)
    :return:
    """
    d2 = second_derivative_matrix_nes(x, scale_free=True)
    derv_1 = first_derv_nes_cvxpy(x, y_var)

    if l_norm == 2:
        norm = cvxpy.sum_squares
    else:
        norm = cvxpy.norm1

    reg_0 = alpha_0 * norm(y_var)
    reg_1 = alpha_1 * norm(derv_1)
    reg_2 = alpha_2 * norm(d2 * y_var)
    reg = reg_0 + reg_1 + reg_2
    return reg


def get_obj_func_model(y, y_err=None, monotonic=False):
    """
    Get the objective function and the model as cvxpy expressions
    :param y: The y variable, numpy array
    :param y_err: The y_err variable, numpy array
        Default to 1
    :param monotonic: If set to True, will result in a monotonically
        increasing function. Default is False.
    :return: objective function and the model
    """
    n = len(y)
    if y_err is None:
        y_err = np.ones(n)
    else:
        assert len(y_err) == n

    y_var = cvxpy.Variable(n, pos=monotonic)

    buff = 0.01 * np.median(abs(y))
    buff_2 = buff ** 2
    isig = 1 / np.sqrt(buff_2 + y_err ** 2)

    c_matrix = cumulative_matrix(n)
    model = c_matrix * y_var

    diff = cvxpy.multiply(isig, model - y)
    obj_func = cvxpy.sum(cvxpy.huber(diff))
    return obj_func, model

import numpy as np
import cvxpy
from scipy.interpolate import interp1d
from trendfilter.derivatives import second_derivative_matrix_nes, \
    first_derv_nes_cvxpy, cumulative_matrix


def trend_filter(x, y, y_err=None, alpha_2=0.0,
                 alpha_1=0.0, alpha_0=0.0, l_norm=2,
                 constrain_zero=False, monotonic=False,
                 linear_deviations=None):
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
    :param linear_deviations: list of linear deviation objects
    :return: The fit model information
    """

    if linear_deviations is None:
        linear_deviations = []

    assert l_norm in [1, 2]
    n = len(x)

    # get the y_err is not supplied
    assert len(y) == n
    if y_err is None:
        y_err = np.ones(n)
    else:
        assert len(y_err) == n

    # the objective function
    result = get_obj_func_model(y, y_err=y_err,
                                monotonic=monotonic,
                                linear_deviations=linear_deviations)

    # TODO: this seems wrong
    y_var = result['objective_function'].variables()[0]
    # y_var = result['base_model']

    # the regularization
    reg_sum, regs = get_reg(x, y_var, l_norm, alpha_0, alpha_1, alpha_2,
                            linear_deviations=linear_deviations)

    # the total objective function with regularization
    obj_total = result['objective_function'] + reg_sum

    # The objective
    obj = cvxpy.Minimize(obj_total)

    # Get the constraints if any
    constraints = []
    if constrain_zero:
        constraints.append(result['model'][0] == 0)

    # define and solve the problem
    problem = cvxpy.Problem(obj, constraints=constraints)
    problem.solve()

    # return the model fit values
    y_fit = result['model'].value

    tf_result = {'function': interp1d(x, y_fit, fill_value="extrapolate"),
                 'model': result['model'],
                 'base_model': result['base_model'],
                 'objective_model': result['objective_function'],
                 'regularization_total': reg_sum,
                 'regularizations': regs,
                 'objective_total': obj,
                 'y_fit': result['model'].value,
                 'constraints': constraints}

    return tf_result


def get_reg(x, base_model, l_norm, alpha_0, alpha_1, alpha_2, linear_deviations=None):
    """
    Get the regularization term
    :param x: The x-value, numpy array
    :param base_model: The y variable, cvxpy.Variable(n)
    :param l_norm: 1 or 2 to use either L1 or L2 norm
    :param alpha_0: Regularization against non-zero slope (first derivative)
        Setting this very high will create a series of steps (if L1)
    :param alpha_1: Regularization against non-constant slope (second derivative)
        Setting this very high will result in piecewise linear model (if L1)
    :param alpha_2: Regularization against (third derivative)
        Setting this very high will result in piecewise quadratic (if L1)
    :param linear_deviations: list of linear deviation objects
    :return: (sum of regs, list of regs)
    """
    d2 = second_derivative_matrix_nes(x, scale_free=True)
    derv_1 = first_derv_nes_cvxpy(x, base_model)

    if l_norm == 2:
        norm = cvxpy.sum_squares
    else:
        norm = cvxpy.norm1

    reg_0 = alpha_0 * norm(base_model)
    reg_1 = alpha_1 * norm(derv_1)
    reg_2 = alpha_2 * norm(d2 @ base_model)
    regs = [reg_0,  reg_1,  reg_2]

    for lin_dev in linear_deviations:
        reg = lin_dev['alpha'] * norm(lin_dev['variable'])
        regs.append(reg)

    reg_sum = sum(regs)

    return reg_sum, regs


def get_obj_func_model(y, y_err=None, monotonic=False, linear_deviations=None):
    """
    Get the objective function and the model as cvxpy expressions
    :param y: The y variable, numpy array
    :param y_err: The y_err variable, numpy array
        Default to 1
    :param monotonic: If set to True, will result in a monotonically
        increasing function. Default is False.
    :param linear_deviations: List of completed linear deviation objects
    :return: objective function and the model
    """

    if linear_deviations is None:
        linear_deviations = []

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
    base_model = c_matrix @ y_var
    model = base_model

    for lin_dev in linear_deviations:
        model += lin_dev['model_contribution']

    diff = cvxpy.multiply(isig, model - y)
    obj_func = cvxpy.sum(cvxpy.huber(diff))

    result = {'base_model': base_model,
              'model': model,
              'objective_function': obj_func}

    return result

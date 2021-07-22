# trendfilter: 

Trend filtering is about building a model for a 1D time series
that has some nice properties such as smoothness or sparse 
changes in slopes (piecewise linear).

The objective to be minimized is, in our case, Huber loss with
regularization on 1st, 2nd or 3rd derivative plus some constraints. 
Can be either L1 or L2 norms for regularization.

This library provides a flexible and powerful 
python function to do this and is built on top of the 
cvxpy optimization library.

# Install

pip install trendfilter

or clone the repo. 

# Examples: 

Contruct some x, y data where there is noise as well as few 
outliers. See 
[test file](https://github.com/dave31415/trendfilter/blob/master/test/test_mono.py)
for prep_data code and plotting code.

First build the base model with no regularization. This is essentially
just an overly complex way of constructing an interpolation function. 
If you use the keyword return_function, it returns an actual 
function which in interpolation of the model. Without that, 
it return an array, evaluated at the set of x points. 

```
x, y_noisy = prep_data()
y_fit = trend_filter(x, y_noisy)
```

![BaseModel](./plots/bokeh_plot_base_model_no_reg.png)

This has no real use by itself. It's just saying the model is the
same as the data points.

You'll always want to give it some key-words to
apply some kind of regularization or constraint so that the model
is not simply the same as the noisy data.

Let's do something more useful. Don't do any regularization yet. Just
force the model to be monotonically increasing. So basically, it 
is looking for the model which has minimal Huber loss but is 
constrained to be monotonically increasing. 

We use Huber loss because it is robust to outliers while still
looking like quadratic loss for small values.

```
y_fit = trend_filter(x, y_noisy, monotonic=True)
```

![MonoModel](./plots/bokeh_plot_best_mono.png)

The green line, by the way, just shows that the function can be extrapolated 
which is a very useful thing, for example, if you want to make predictions
about the future.

Ok, now let's do an L1 trend filter model. So we are going to 
penalize any non-zero second derivative with an L1 norm. As we 
probably know, L1 norms induce sparseness. So the second dervative at 
most of the points will be exactly zero but will probably be non-zero
at a few of them. Thus, we expect piecewise linear trends that 
occasionally have sudden slope changes.

```
y_fit = trend_filter(x, y_noisy, l_norm=1, alpha_1=0.2)
```

![L1TF](./plots/bokeh_plot_l1_trend_filter.png)

Let's do the same thing but enforce it to be monotonic.

```
y_fit = trend_filter(x, y_noisy, l_norm=1, alpha_1=0.2, monotonic=True)
```

![L1TFMono](./plots/bokeh_plot_l1_trend_filter_mono.png)


Now let's increase the regularization parameter to give a higher
penalty to slope changes. It results in longer trends. Fewer slope
changes. Overall, less complexity.

```
y_fit = trend_filter(x, y_noisy, l_norm=1, alpha_1=2.0)
```

![L1TFMoreReg](./plots/bokeh_plot_l1_trend_filter_more_reg.png)


Did you like the stair steps? Let's do that again. But now
we will not force it to be monotonic. We are going to put an 
L1 norm on the first derivative. This produces a similar 
output but it could actually decrease if the data actually
did so. 

```
y_fit = trend_filter(x, y_noisy, l_norm=1, alpha_0=8.0)
```

![L1TFSteps](./plots/bokeh_plot_stair_steps.png)


Let's do L2 norms for regularization on the second 
derivative. L2 norms don't care very much about small values. 
They don't force them all the way to zero to create sparse 
solution. They care more about big values. This results is a 
smooth continuous curve. This is a nice way of doing robust 
smoothing. 

```
y_fit = trend_filter(x, y_noisy, l_norm=2, alpha_1=1.0)
```

![L2TF](./plots/bokeh_plot_l2_trend_filter.png)

Let's use L1 norm again but put it on the 3rd derivative. 
This will result in sparse changes in second derivative. So
locally things will be quadratic but with sparse changes. 

We also have a key-word to constraint the function to pass 
through the origin, (0,0).

```
y_fit = trend_filter(x, y_noisy, l_norm=1, alpha_2=3.0, constrain_zero=True)
```

![L2PWQ](./plots/bokeh_plot_piecewise_quadratic_constrain_zero.png)


Here is the full function signature.

```
def trend_filter(x, y, y_err=None, alpha_2=0.0,
                 alpha_1=0.0, alpha_0=0.0, l_norm=2,
                 constrain_zero=False, monotonic=False,
                 return_function=False):
```

So you see there are alpha key-words for regularization parameters. 
The number n, tells you the n+1 derivative is being penalized.
You can use any, all or none of them. The key-word
l_norm gives you the choice of 1 or 2. Monotonic and 
constrain_zero, we've explained already. 

The return_function allows you to get either the model at 
the points in an array (default) or the interpolating 
function of the model to apply to some other x values. 

We didn't discuss y_err. That's the uncertainty on y. The
default is 1. The Huber loss is actually applied to (data-model)/y_err.
So, you can weight points differently if you have a good reason to,
such as knowing the actual error values or knowing that some points 
are dubious or if some points are known exactly. That would be the limit
as y_err goes towards zero and it will make the curve go through those
points.

All in all, these keywords give you a huge amount of freedom in
what you want your model to look like. 

Enjoy!




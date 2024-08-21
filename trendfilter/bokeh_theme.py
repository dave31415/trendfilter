"""
Set the Bokeh plotting defaults
"""

from bokeh.io import curdoc
from bokeh.themes import Theme

curdoc().theme = Theme(filename="bokeh_theme.yaml")


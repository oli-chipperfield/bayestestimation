import numpy as np
import arviz as az
import plotly.graph_objects as go

from bayestestimation.bayesthelpers import _calculate_kde, _calculate_map


def _get_centre_lines(draws, method):
    # Derives map or median or mean for plotting purposes   
    if method == 'hdi':
        cl = _calculate_map(draws)
    elif method == 'quantile':
        cl = np.quantile(draws, 0.5)
    x, kde_density = _calculate_kde(draws, num=100)
    best_y = kde_density[np.argmin(np.abs(x - cl))]
    return {'x': cl, 'y': best_y}


def _get_intervals(draws, method, bounds):
    # Derives HDI or credible intervals for plotting purposes
    if method == 'hdi':
        il = az.hdi(draws, bounds)
    elif method == 'quantile':
        il = np.quantile(draws, bounds)
    x, kde_density = _calculate_kde(draws, num=100)
    subx = x[(x > il[0]) & (x < il[1])]
    kde_density = kde_density[(x > il[0]) & (x < il[1])]
    return {'x': subx, 'y': kde_density}


def _make_density_go(draws, name, col='#000000'):
    # Makes a KDE density graph object
    x, kde_density = _calculate_kde(draws, num=100)
    graphobj = go.Scatter(x=x,
                          y=kde_density,
                          line={'color': col},
                          name=name)
    return graphobj


def _make_histogram_go(draws, name, col='#000000'):
    # Makes a histogram graph object
    graphobj = go.Histogram(x=draws, 
                            nbinsx=40,
                            opacity=0.2, 
                            histnorm='probability density',
                            marker_color=col,
                            name=name)
    return graphobj


def _make_area_go(kde_object, name, col='#000000'):
    # Makes area under curve graph object
    graphobj = go.Scatter(x=kde_object['x'],
                          y=kde_object['y'],
                          fill='tozeroy',
                          line={'color': col},
                          opacity=0.5,
                          name=name)
    return graphobj


def _make_line_go(line_object, name, col='#000000'):
    # Make line graph object
    graphobj = go.Scatter(x=[line_object['x'], line_object['x']],
                          y=[0, line_object['y']],
                          mode='lines',
                          line={'color': col, 'dash': 'dash'},
                          name=name)
    return graphobj


def _make_delta_line(draws, delta_line, col='#d62728'):
    # Make line dictionary object for the delta posterior
    x, kde_density = _calculate_kde(draws, num=100)
    maxy = np.max(kde_density) * 1.2    
    linedict = {'type': 'line',
                'x0': delta_line,
                'x1': delta_line,
                'y0': 0,
                'y1': maxy,
                'xref': 'x3',
                'yref': 'y3',
                'line': dict(color=col, dash='dot')}
    return linedict
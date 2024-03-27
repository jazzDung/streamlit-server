import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union
from configs.config import *


def line_chart_config(
        fig,
        line_width: float = 2,
        marker_size: Union[float, None] = None,
        axis_title_size: float = 12,
        x_axis_data_type: str = 'numeric',
        y_axis_data_type: str = 'numeric',
        tool_tip_font_size: float = 12,
        legend_position: str = 'top',
):
    fig = set_line_width(fig, line_width)
    fig = add_marker(fig, marker_size) if marker_size is not None else fig
    fig = set_axis_title_size(fig, axis_title_size)
    fig = set_axis_title_size(fig, axis_title_size)
    fig = set_tool_tip_font_size(fig, tool_tip_font_size)
    fig = change_background_color(fig)
    fig = disable_zoom(fig)
    fig.update_xaxes(
        tickfont=dict(color=LIGHT_BLACK)
    )
    fig.update_yaxes(
        tickfont=dict(color=LIGHT_BLACK)
    )
    fig.update_layout(
        margin=dict(
            t=10,
            b=20,
            pad=5
        ),
    )
    if x_axis_data_type == 'percentage':
        fig = axis_percentage(fig, 'x')
    if y_axis_data_type == 'percentage':
        fig = axis_percentage(fig, 'y')
    if legend_position == 'top':
        fig = set_legend_at_top(fig)
    return fig


def set_axis_tick_color(fig, color=LIGHT_BLACK):
    fig.update_xaxes(
        tickfont=dict(color=color)
    )

    fig.update_yaxes(
        tickfont=dict(color=color)
    )
    return fig


def bar_chart_config(fig):
    return fig


def change_background_color(fig, color=BACKGROUND_COLOR):
    return fig.update_layout(
        plot_bgcolor=color,
        paper_bgcolor=color,
    )


def disable_zoom(fig):
    return fig.update_layout({
        'yaxis': {'fixedrange': True},
        'xaxis': {'fixedrange': True},
    })


def axis_percentage(fig, axis: str):
    if axis == 'x':
        fig.update_xaxes(
            tickformat=',.0%',
            range=[0, 1],
            tickfont=dict(color=LIGHT_BLACK)
        )
    elif axis == 'y':
        fig.update_yaxes(
            tickformat=',.0%',
            range=[0, 1],
            tickfont=dict(color=LIGHT_BLACK)
        )
    else:
        logging.warning('Invalid axis name')

    return fig


def set_line_width(fig, width=3):
    fig.update_traces(
        line=dict(width=width)
    )
    return fig


def add_marker(fig, size=10):
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=size),
        mode="markers+lines"
    )
    return fig


def set_axis_title_size(fig, size=16):
    fig.update_xaxes(
        title_font=dict(size=size),
        tickfont=dict(size=size),
        title_font_color=LIGHT_BLACK,
    ).update_yaxes(
        title_font=dict(size=size),
        tickfont=dict(size=size),
        title_font_color=LIGHT_BLACK,
    )
    return fig


def set_legend_at_top(fig):
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


def set_tool_tip_font_size(fig, size=25):
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(font=dict(size=size))
    )
    return fig


def double_axis_line_chart(
        x: Union[list, pd.Series],
        first_y: list,
        second_y: list,
        y_label: list,
        first_y_format: Union[str, None] = None,
        first_y_range: Union[list, None] = None,
        second_y_format: Union[str, None] = None,
        second_y_range: Union[list, None] = None,
        colors: Union[list, None] = None,
        line_shape: str = 'spline'
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if isinstance(x, list) == list and len(x) != len(first_y) + len(second_y):
        logging.warning('Amount of x axis fields and y axis fields not equal')

    if len(first_y) + len(second_y) != len(y_label):
        logging.warning('Amount of y axis lable and y axis fields not equal')

    if isinstance(x, pd.Series):
        x = [x for i in range(len(first_y + second_y))]

    for i in range(len(first_y)):
        fig.add_trace(
            go.Scatter(
                x=x[i],
                y=first_y[i],
                line=dict(color=colors[i]),
                line_shape=line_shape,
                name=y_label[i]),

            secondary_y=False,
        )

    x = x[len(first_y):]
    y_label = y_label[len(first_y):]
    colors = colors[len(first_y):]

    for i in range(len(second_y)):
        fig.add_trace(
            go.Scatter(
                x=x[i],
                y=second_y[i],
                line=dict(color=colors[i]),
                line_shape=line_shape,
                name=y_label[i]),
            secondary_y=True,
        )

    fig.update_layout(
        yaxis={
            'tickformat': first_y_format,
            'range': first_y_range
        },
        yaxis2={
            'tickformat': second_y_format,
            'range': second_y_range
        },
    )
    return fig

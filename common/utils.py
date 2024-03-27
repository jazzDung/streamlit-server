import logging
from common.common import TextRender
from common.template_loader import TemplateLoader
from configs.config import *

TEMPLATE_LOADER = TemplateLoader(TEMPLATE_LOC)


def color_decider(value, threshold=None, threshold_color=None):
    if threshold is None:
        threshold = [0]
    if threshold_color is None:
        threshold_color = [LIGHT_RED, DARK_CYAN]
    if len(threshold) + 1 != len(threshold_color):
        logging.warning('Amount of color and threshold not match, some threshold will have same color')

    color = threshold_color[0]
    for i in range(len(threshold)):
        if value >= threshold[i]:
            color = threshold_color[min(len(threshold_color) - 1, i + 1)]
    return color


def gen_percentage_string(value, prefix: str = '', suffix: str = ''):
    render = TextRender({'value': value})
    return prefix + render.render('{{ (value) | f_percent }}') + suffix


def gen_currency_string(value, prefix: str = '', suffix: str = ''):
    render = TextRender({'value': value})
    return prefix + render.render('{{ value | vn_currency }}') + suffix


def gen_title_and_value_same_line(
        cell,
        title: str,
        value: str,
        color: str = None,
        loader=TEMPLATE_LOADER
):
    args = {
        'title': title,
        'value': value,
        'color': color,
    }
    html = loader.render("title_and_value.tpl", **args)
    return cell.markdown(html, unsafe_allow_html=True)


def gen_big_number(
        cell,
        title: str,
        value: str,
        color: str = None,
        loader=TEMPLATE_LOADER
):
    args = {
        'title': title,
        'value': value,
        'color': color,
    }
    html = loader.render("big_number.tpl", **args)
    return cell.markdown(html, unsafe_allow_html=True)
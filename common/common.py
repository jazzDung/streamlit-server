from jinja2 import Environment, meta
from babel.numbers import format_decimal
import random
import math
import logging


class TextRender:
    @classmethod
    def f_vn_currency(cls, value):
        if value < 0:
            return "âm " + format_decimal(abs(value) / 1000000000, format='#,##0.##;-#', locale='vi_VN') + " tỷ đồng"
        return format_decimal(value / 1000000000, format='#,##0.##;-#', locale='vi_VN') + " tỷ đồng"

    @classmethod
    def vn_currency(cls, value):
        if value < 0:
            return format_decimal(abs(value) / 1000000000, format='#,##0.##;-#', locale='vi_VN') + " tỷ đồng"
        return format_decimal(value / 1000000000, format='#,##0.##;-#', locale='vi_VN') + " tỷ đồng"

    @classmethod
    def f_vn_int_currency(cls, value):
        if value < 0:
            return format_decimal(abs(value) / 1000000000, format='#,##0;-#', locale='vi_VN') + " tỷ đồng"
        return format_decimal(value / 1000000000, format='#,##0;-#', locale='vi_VN') + " tỷ đồng"

    @classmethod
    def f_percent(cls, value):
        return format_decimal(value * 100, format='#,##0.##;-#', locale='vi_VN') + "%"

    @classmethod
    def vn_currency_giacophieu(cls, value):
        value_int = int(value * 1000)
        return format(value_int, ',d').replace(',', '.')

    def __init__(self, input_data) -> None:
        assert isinstance(input_data, dict)
        self.input_data = input_data
        self.env = Environment()
        self.env.filters['f_vn_currency'] = self.f_vn_currency
        self.env.filters['vn_currency'] = self.vn_currency
        self.env.filters['f_vn_int_currency'] = self.f_vn_int_currency
        self.env.filters['f_percent'] = self.f_percent
        self.env.filters['vn_currency_giacophieu'] = self.vn_currency_giacophieu

    def render(self, tpl, visible_cond_func=lambda x: True):
        parsed_content = self.env.parse(tpl)
        variables = meta.find_undeclared_variables(parsed_content)
        for variable in variables:
            if not self.check_exists(variable, self.input_data):
                logging.warning("Variable {} not found in input_data or nan/None".format(variable))
                return ""

        if visible_cond_func(self.input_data):
            return self.env.from_string(tpl).render(self.input_data)
        else:
            return ""

    def render_multi_parts(self, tpl_parts, visible_cond_func=lambda x: True):
        assert isinstance(tpl_parts, list)
        if visible_cond_func(self.input_data):
            if len(tpl_parts) == 0:
                return ""
            if isinstance(tpl_parts[0], tuple):
                parts = [self.render(x[0], x[1]) for x in tpl_parts]
            else:
                parts = [self.render(x) for x in tpl_parts]
            return "".join(parts)
        else:
            return ""

    def rand_render(self, tpls, visible_cond_func=lambda x: True):
        assert isinstance(tpls, list)
        if visible_cond_func(self.input_data):
            tpl = random.choice(tpls)
            return self.render(tpl)
        else:
            return ""

    def rand_choice(self, choices, fallback):
        assert isinstance(choices, list)
        assert fallback is not None

        tmps = [x for x in choices if x is not None]
        if len(tmps) == 0:
            return self.render(fallback)
        return random.choice(tmps)

    @classmethod
    def check_exists(cls, key, data):
        if key in data and data[key] is not None:
            if isinstance(data[key], str):
                return len(data[key]) > 0
            else:
                return not math.isnan(data[key])
        else:
            return False

import os
from datetime import timezone, timedelta


DSAI_TIMESCALEDB_PRD = "postgresql://postgres:nIV0JOWbNLpZuFk0lHkq1uF02AKvIryLAV9USAvkvLT9AAcoIbFK0ydZGabETOKK@103.151.242.52:5432/dsai"
# DSAI_TIMESCALEDB_PRD = "postgresql://dsai:nTUUj0pFft8AqzzhdmHIHGj3H4x9lqJGX74Q7mwcQ6us4Qo5ANMG4H6kryTfJWt3@encapital-dsai-timescaledb.dsai:5432/dsai"
REDSHIFT_PRD = "redshift+psycopg2://dungpham:DJQc6sqEJxPyFExf@encapital-redshift-data-warehouse.cir6kkgprzvy.ap-southeast-1.redshift.amazonaws.com:5439/landing"
GET_SYMBOL_EFFECT_VNI_URL = "https://services.entrade.com.vn/market-api/basket-influence?type=vnindex"
GET_CURRENT_VNINDEX_URL = "https://services.entrade.com.vn/chart-api/snapshots/index/current?symbols=VNINDEX"
GET_TOP_SYMBOL_GAINERS_URL = "https://services.entrade.com.vn/stock-transformer/statistics/gainers?tradingValue=0&size=10"
GET_TOP_SYMBOL_LOSER_URL = "https://services.entrade.com.vn/stock-transformer/statistics/losers?=&tradingValue=0&size=10"
TEMPLATE_LOC = 'template'

CONFIG = {
    'displayModeBar': False
}

BACKGROUND_COLOR = 'rgba(23, 26, 39, 0)'
DARK_CYAN = 'rgb(0, 161, 118)'
CYAN = 'rgb(0, 204, 150)'
LIGHT_CYAN = 'rgb(149, 230, 208)'
LIGHT_RED = 'rgb(239, 58, 76)'
SUPER_LIGHT_RED = 'rgb(255, 160, 160)'
DNSE_RED = "rgba(145,18,28,100)"
LIGHT_GRAY = 'rgb(211, 211, 211)'
LIGHT_BLACK = 'rgb(26, 17, 16)'
BLACK = 'rgb(0, 0, 0)'
BACKGROUND_COLOR_LIGHT = 'rgb(255, 233, 243)'
AMBER = 'rgb(218, 165, 32)'
INVISIBLE = 'rgba(0, 0, 0, 0)'
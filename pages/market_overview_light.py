import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import requests
import streamlit as st
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta, date

from common.plotly_utils import *
from common.utils import *
from configs.config import *
from database.utils import *

dsai_engine = create_engine(DSAI_TIMESCALEDB_PRD)
redshift_engine = create_engine(REDSHIFT_PRD)

# App title
st.set_page_config(page_title="Market Overview", page_icon="📈", initial_sidebar_state="collapsed", layout="wide", )

# Styling
with open('css/market_overview.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache_data
def load_ma_statistic_range(start, end):
    df = get_record_range(
        schema='dwh_market',
        table='fact_period_market_statistic_daily',
        start_date=start,
        end_date=end,
        engine=redshift_engine,
        timestamp_column='txn_date'
    )

    df.replace(
        to_replace=[np.inf, -np.inf, np.nan],
        value=None,
        inplace=True
    )

    df = df.assign(
        down_ma20_rate=df['down_ma20'] / (df['down_ma20'] + df['up_ma20']),
        down_ma50_rate=df['down_ma50'] / (df['down_ma50'] + df['up_ma50']),
        down_ma100_rate=df['down_ma100'] / (df['down_ma100'] + df['up_ma100']),
        down_ma200_rate=df['down_ma200'] / (df['down_ma200'] + df['up_ma200']),
    )

    return df


@st.cache_data
def load_fact_period_trading_value_ratio_daily_range(start, end):
    df = get_record_range(
        schema='dwh',
        table='fact_period_trading_value_ratio_daily',
        start_date=start,
        end_date=end,
        engine=redshift_engine,
        timestamp_column='txn_date'
    )

    df = df.groupby(['trader_type', 'txn_date'], as_index=False).agg(
        buy_value=('buy_value', 'sum'),
        sell_value=('sell_value', 'sum'),
    )

    df[['sell_value']] = df[['sell_value']] * (-1)
    return df


def load_sector_metric(start, end):
    start_date_string = datetime.strftime(start, '%Y-%m-%d')
    end_date_string = datetime.strftime(end, '%Y-%m-%d')
    sql = f"""
        select
            *
        from dwh_market.fact_period_sector_metric_daily
        where
            ngay::date >= '{start_date_string}'
            and ngay::date <= '{end_date_string}'
            and is_holiday = 'N'
            and cap = 3
        order by  ngay desc;
        """
    df = pd.read_sql_query(
        sql=sql,
        con=redshift_engine
    )
    return df


@st.cache_data
def load_market_price_metric_range(start, end):
    start_date_string = datetime.strftime(start, '%Y-%m-%d')
    end_date_string = datetime.strftime(end, '%Y-%m-%d')
    sql = f"""
            select
                indexed_timestamp_
                , avg(pha_day_1m) as pha_day_1m
                , avg(vuot_dinh_1m) as vuot_dinh_1m
                , avg(pha_day_3m) as pha_day_3m
                , avg(vuot_dinh_3m) as vuot_dinh_3m
            from dsai.dnse_thong_ke_cuoi_ngay_1d0h0m_1
            where 
                indexed_timestamp_::date >= '{start_date_string}'
                and indexed_timestamp_::date <= '{end_date_string}'
            group by indexed_timestamp_
            order by  indexed_timestamp_ desc
        """
    df = pd.read_sql_query(
        sql=sql,
        con=redshift_engine
    )
    return df


@st.cache_data
def load_vnindex_point_range(start, end):
    df = get_record_range(
        schema='dwh_staging',
        table='int_index_ohlc_days',
        start_date=start,
        end_date=end,
        engine=redshift_engine,
        timestamp_column='txn_date'
    )

    df = df[df['symbol'] == 'VNINDEX']
    return df


# Global filter

con0 = st.container()
c1, padding, c2 = con0.columns([1, 0.1, 1])
c1.header('Filter dữ liệu')

ma_chart_time = c1.slider(
    label='Thời gian',
    min_value=date.today(),
    max_value=date.today() - timedelta(days=100),
    value=(date.today() - timedelta(days=30), date.today()),
    format="MM/DD/YY",
    # label_visibility='hidden'
)

# Tổng quan VNINDEX
con0 = st.container()
c1, padding, c2 = con0.columns([1, 0.1, 1])

c1.header(f'Tỉ lệ thanh khoản theo thời gian')
start_date, end_date = ma_chart_time
sector_metric = load_sector_metric(start=start_date, end=end_date)
top_10_sector = sector_metric.groupby(by=['ten_nganh']).agg({'total_thanh_khoan': 'sum'}).sort_values(by='total_thanh_khoan', ascending=False).head(10)
sector_metric = sector_metric[ sector_metric['ten_nganh'].isin(top_10_sector.index)]
sector_metric_total_cap = sector_metric.groupby(by=['ngay']).agg({'total_thanh_khoan': 'sum'})
sector_metric = pd.merge(sector_metric, sector_metric_total_cap, on=['ngay','ngay'], suffixes=['', '_all'])
sector_metric['sector_liquidity_ratio'] = sector_metric['total_thanh_khoan'] / sector_metric['total_thanh_khoan_all']
sector_metric = sector_metric.sort_values(by='total_thanh_khoan')

fig_liquidity = px.area(
    data_frame=sector_metric,
    x="ngay",
    y="sector_liquidity_ratio",
    color="ten_nganh",
    markers=True,
    labels={"ngay": "Ngày", "sector_liquidity_ratio": "Tỉ lệ thanh khoản", "ten_nganh": "Ngành"},
    height=600
)

fig_liquidity.update_traces(
    hovertemplate='<br>Tỉ lệ thanh khoản: %{y}',
)

config = {'displayModeBar': False}

fig_liquidity = line_chart_config(
    fig=fig_liquidity,
    marker_size=8,
    axis_title_size=16,
    tool_tip_font_size=18,
    y_axis_data_type='percentage'
)


fig_liquidity.update_layout(
    hovermode="x unified",
    yaxis=dict(hoverformat=',.0%'),
    font_color=BLACK,
)

c1.plotly_chart(fig_liquidity, config=config, use_container_width=True)

c2.header(f'Thanh khoản thị trường theo thời gian')
# t1, t2 = c2.tabs(["📈 Chart", "🗃 Data"])
start_date, end_date = ma_chart_time

sector_metric = load_sector_metric(start=start_date, end=end_date)
top_10_sector = sector_metric.groupby(by=['ten_nganh']).agg({'total_thanh_khoan': 'sum'}).sort_values(by='total_thanh_khoan', ascending=False).head(10)
sector_metric = sector_metric[sector_metric['ten_nganh'].isin(top_10_sector.index)]
sector_metric = sector_metric.sort_values(by='total_thanh_khoan')

fig_liquidity = px.area(
    data_frame=sector_metric,
    x="ngay",
    y="total_thanh_khoan",
    color="ten_nganh",
    markers=True,
    labels={"ngay": "", "total_thanh_khoan": "", "ten_nganh": "Ngành"},
    height=600
)

fig_liquidity = line_chart_config(
    fig=fig_liquidity,
    marker_size=8,
    axis_title_size=16,
    tool_tip_font_size=18
)

fig_liquidity.update_traces(
    hovertemplate='<br>Thanh khoản: %{y}',
)

fig_liquidity.update_layout(
    hovermode="x unified",
)

config = {'displayModeBar': False}


c2.plotly_chart(fig_liquidity, config=config, use_container_width=True)
# t1.plotly_chart(fig_liquidity, config=config, use_container_width=True)
# t2.dataframe(all_liquidity_same_period.set_index('indexed_timestamp_'))

# Mã dưới đường MA
c2.header('Tỉ lệ mã dưới đường MA theo thời gian')
# t1, t2 = c2.tabs(["📈 Chart", "🗃 Data"])

start_date, end_date = ma_chart_time
ma_statistic_range = load_ma_statistic_range(start_date, end_date)
vnindex_point_range = load_vnindex_point_range(start_date, end_date)

fig_under_ma = double_axis_line_chart(
    x=ma_statistic_range['txn_date'],
    first_y=[
        ma_statistic_range['down_ma20_rate'],
        ma_statistic_range['down_ma50_rate'],
        ma_statistic_range['down_ma100_rate'],
        ma_statistic_range['down_ma200_rate'],
    ],
    second_y=[vnindex_point_range["close_price"]],
    y_label=[
        'Dưới đường MA20',
        'Dưới đường MA50',
        'Dưới đường MA100',
        'Dưới đường MA200',
        'VNINDEX'
    ],
    first_y_format=',.0%',
    first_y_range=[0, 1],
    colors=['#032174', '#1560BD', '#2C7DA0', '#87CEEB', DNSE_RED]
)

fig = line_chart_config(
    fig=fig_under_ma,
    marker_size=8,
    axis_title_size=16,
    tool_tip_font_size=16,
).update_traces(
    selector={"name": "VNINDEX"},
    line={"dash": "dash"}
).update_layout(
    hovermode="x unified",
    autosize=False,
    height=600,
    yaxis={'fixedrange': True},
    xaxis={'fixedrange': True},
)

# c2.plotly_chart(fig_under_ma, config=config, use_container_width=True)
c2.plotly_chart(fig, config=config, use_container_width=True)

# t1.plotly_chart(fig_under_ma, config=config, use_container_width=True)
# t2.dataframe(ma_statistic_range.set_index('txn_date'))


# Vượt đỉnh phá đáy
market_price_metric_range = load_market_price_metric_range(start_date, end_date)
c1.header(f'Tỉ lệ mã vượt đỉnh phá đáy')

fig_market_price_metric = go.Figure()
fig_market_price_metric.add_trace(
    go.Scatter(
        x=market_price_metric_range["indexed_timestamp_"],
        y=market_price_metric_range["pha_day_1m"],
        customdata=[gen_percentage_string(i) for i in market_price_metric_range["pha_day_1m"]],
        line=dict(width=4, color=LIGHT_RED),
        line_shape="spline",
        hovertemplate="<br>".join([
            "%{customdata}",
        ]),
        name="Tỉ lệ phá đáy 1 tháng",
    )
)

fig_market_price_metric.add_trace(
    go.Scatter(
        x=market_price_metric_range["indexed_timestamp_"],
        y=market_price_metric_range["vuot_dinh_1m"],
        customdata=[gen_percentage_string(i) for i in market_price_metric_range["vuot_dinh_1m"]],
        line=dict(width=4, color=CYAN),
        line_shape="spline",
        hovertemplate="<br>".join([
            "%{customdata}",
        ]),
        name="Tỉ lệ vượt đỉnh 1 tháng",
    )
)

fig_market_price_metric.add_trace(
    go.Scatter(
        x=market_price_metric_range["indexed_timestamp_"],
        y=market_price_metric_range["pha_day_3m"],
        customdata=[gen_percentage_string(i) for i in market_price_metric_range["pha_day_3m"]],
        line=dict(width=4, color=DNSE_RED),
        line_shape="spline",
        hovertemplate="<br>".join([
            "%{customdata}",
        ]),
        name="Tỉ lệ phá đáy 3 tháng",
    )
)

fig_market_price_metric.add_trace(
    go.Scatter(
        x=market_price_metric_range["indexed_timestamp_"],
        y=market_price_metric_range["vuot_dinh_3m"],
        customdata=[gen_percentage_string(i) for i in market_price_metric_range["vuot_dinh_3m"]],
        line=dict(width=4, color=DARK_CYAN),
        line_shape="spline",
        hovertemplate="<br>".join([
            "%{customdata}",
        ]),
        name="Tỉ lệ vượt đỉnh 3 tháng",
    )
)

# fig_market_price_metric.update_traces()

fig_market_price_metric = line_chart_config(
    fig=fig_market_price_metric,
    marker_size=8,
    axis_title_size=16,
    # y_axis_data_type='percentage',
    tool_tip_font_size=18
)

fig_market_price_metric.update_layout(
    height=600,
    hovermode="x unified",
).update_yaxes(
            tickformat=',.0%',
)

c1.plotly_chart(fig_market_price_metric, config=config, use_container_width=True)


# Tương quan thay đổi giá ngành và total cap
sector_metric = load_sector_metric(start=datetime.now() - timedelta(days=1), end=datetime.now() - timedelta(days=1))
sector_metric.fillna(value=0, inplace=True)
top_10_sector = sector_metric.groupby(by=['ten_nganh']).agg({'total_thanh_khoan': 'sum'}).sort_values(by='total_thanh_khoan', ascending=False).head(10)
sector_metric = sector_metric[ sector_metric['ten_nganh'].isin(top_10_sector.index)]


c1.header(f'Tương quan thay đổi về giá ngành và total cap')
fig_top_net_buy = px.scatter(
    data_frame=sector_metric,
    x="gia_nganh_today_vs_5d_ago",
    y="total_cap_change",
    color="ten_nganh",
    size="total_thanh_khoan",
    labels={
        "ten_nganh": "Ngành",
        "gia_nganh_today_vs_5d_ago": "Thay đổi giá ngành",
        "total_cap_change": "Thay đổi cap",
        "total_thanh_khoan": "Thanh khoản"
    },
    height=600
).update_traces(
    marker=dict(
        line=dict(width=3, color='DarkSlateGrey')
    ),
    selector=dict(mode='markers')
).update_xaxes(
    # range=[sector_metric['gia_nganh_today_vs_5d_ago'].min() / 1.1, sector_metric['gia_nganh_today_vs_5d_ago'].max() * 1.1],
    tickfont=dict(color=LIGHT_BLACK)
).update_yaxes(
    # range=[sector_metric['total_cap_change'].min() / 1.02, sector_metric['total_cap_change'].max() * 1.02],
    tickfont=dict(color=LIGHT_BLACK)
).update_layout(
    yaxis={'fixedrange': True},
    xaxis={'fixedrange': True},
)

fig_top_net_buy = set_legend_at_top(fig_top_net_buy)
fig_top_net_buy = change_background_color(fig_top_net_buy)
fig_top_net_buy = set_axis_title_size(fig_top_net_buy)

c1.plotly_chart(fig_top_net_buy, config=config, use_container_width=True)


# Mua bán chủ động
# t1, t2 = c2.tabs(["📈 Chart", "🗃 Data"])
# trader_type = st.sidebar.multiselect(
#     label="Nhóm nhà đầu tư",
#     options=['Cá mập', 'Cừu non', 'Sói già'],
#     default=['Cá mập', 'Cừu non', 'Sói già']
# )
t1, t2 = c2.columns([1, 1.2])


trading_value_range = load_fact_period_trading_value_ratio_daily_range(start_date, end_date)

trader_type = t2.radio(
    "Loại nhà đầu tư",
    ["Cá mập", "Cừu non", "Sói già"],
    index=1,
    horizontal=True
)

trading_value_range = trading_value_range[trading_value_range['trader_type'].isin([trader_type])]
trading_value_range['trade_value_cumsum'] = (trading_value_range['buy_value'] + trading_value_range['sell_value']).cumsum()

t1.header(f'Mua bán chủ động theo nhóm nhà đầu tư')

# fig_trading_value = px.line(
#     data_frame=trading_value_range,
#     x="txn_date",
#     y=["buy_value", "sell_value"],
#     labels={"txn_date": ""},
#     color='trader_type',
#     line_shape="spline",
#     height=600
# )
#
# fig_trading_value = line_chart_config(
#     fig=fig_trading_value,
#     marker_size=10
# )
#
# c2.plotly_chart(fig_trading_value, config=config, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=trading_value_range["txn_date"],
    y=trading_value_range["buy_value"],
    customdata=[gen_currency_string(i) for i in trading_value_range["buy_value"]],
    line=dict(color=DARK_CYAN, width=3),
    fill='tozeroy',
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(0, 204, 150, 0.7)',
    # line=dict(width=0),
    hovertemplate="<br>".join([
        "%{customdata}",
    ]),
    name="Khối lượng mua ròng"
))
fig.add_trace(go.Scatter(
    x=trading_value_range["txn_date"],
    y=trading_value_range["sell_value"],
    line=dict(color=DNSE_RED, width=3),
    customdata=[gen_currency_string(i) for i in trading_value_range["sell_value"]],
    fill='tozeroy',
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(239, 58, 76, 0.7)',
    hovertemplate="<br>".join([
        "%{customdata}",
    ]),
    name="Khối lượng bán ròng"
))
fig.add_trace(go.Scatter(
    x=trading_value_range["txn_date"],
    y=trading_value_range["trade_value_cumsum"],
    customdata=[gen_currency_string(i) for i in trading_value_range["trade_value_cumsum"]],
    line_shape="spline",
    mode='lines',
    line=dict(width=4, color=AMBER, dash="dash"),
    marker=dict(size=8),
    # text=[gen_currency_string(i) for i in trading_value_range["trade_value_cumsum"]],
    textposition="top center",
    hovertemplate="<br>".join([
        "%{customdata}"
    ]),
    name="Khối lượng mua bán ròng cộng dồn"
))

fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=20,
    ),

    hovermode="x unified",
    autosize=False,
    height=600,

    font=dict(
        size=16
    ),
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
    yaxis={'fixedrange': True},
    xaxis={'fixedrange': True},
    plot_bgcolor=BACKGROUND_COLOR,
    paper_bgcolor=BACKGROUND_COLOR,
)

fig.update_xaxes(
    tickfont=dict(color=LIGHT_BLACK)
)

fig.update_yaxes(
    tickfont=dict(color=LIGHT_BLACK)
)
fig = set_legend_at_top(fig)

# fig.update_yaxes(
#     range=[min(trading_value_range.buy_value.min(), trading_value_range.change.min()) - 2,
#            max(trading_value_range.max_change.max(), trading_value_range.change.max()) + 2]
# )


fig = set_axis_title_size(fig)

c2.plotly_chart(fig, config=CONFIG, use_container_width=True)
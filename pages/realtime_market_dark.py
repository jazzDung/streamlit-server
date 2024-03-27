import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import time

from plotly.graph_objs import Layout
from sqlalchemy import create_engine
from streamlit.components.v1 import html
from common.plotly_utils import *
from common.utils import *
from configs.config import *
from database.utils import *

dsai_engine = create_engine(DSAI_TIMESCALEDB_PRD)
redshift_engine = create_engine(REDSHIFT_PRD)
# App title
st.set_page_config(page_title="Market Overview", page_icon="📈", initial_sidebar_state="collapsed", layout="wide", )

# Styling
with open('css/realtime_market_dark.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Thanh khoan thi truong
@st.cache_data
def load_lastest_liquidity():
    return get_latest_record(schema='public', table='liquidity_all_resolution', engine=dsai_engine)


@st.cache_data
def load_last_liquidity_of_day():
    return get_last_record_of_day(schema='public', table='liquidity_all_resolution', engine=dsai_engine)


@st.cache_data
def liquidity_same_period():
    return get_last_record_of_day(schema='public', table='liquidity_all_resolution', engine=dsai_engine)


@st.cache_data
def load_latest_price_metric():
    return get_latest_record(schema='public', table='price_metric_all_resolution', engine=dsai_engine)


# VNINDEX stats
@st.cache_data
def get_current_vnindex(url):
    try:
        input_data = requests.get(url).json()
        vni_new = input_data['marketIndices'][0]['indexValue']
        vni_old = input_data['marketIndices'][0]['priorMarketIndex']
        vni_change_percent = 0 if vni_old == 0 else (vni_new / vni_old) - 1
        vni_change_value = abs(vni_new - vni_old)

        return vni_new, vni_old, vni_change_value, vni_change_percent
    except Exception as e:
        logging.warning("Error retrieve data from GET_CURRENT_VNINDEX_URL API")
        logging.warning(e)


def get_vnindex_latest_day(time_grain):
    sql = f"""
        select
            close
            , NVL(close - lag(close, 1) over (partition by symbol, resolution, time::date order by time), 0) as change
            , extract(hour from time + interval '7 hours') as hour
            , extract(minute from time + interval '7 hours') as minute
        from market_.index_ohlc
        where symbol = 'VNINDEX'
        and resolution = '{time_grain}'
        and time::date = (select max(time) from market_.index_ohlc)::date
        order by hour , minute;
        """
    df = pd.read_sql_query(
        sql=sql,
        con=redshift_engine
    )

    return df


def get_min_max_vnindex_point_last_5_days():
    sql = f"""
        with
            max_timestamp as (
                select max(time) as indexed_timestamp_ from market_.index_ohlc
            )
            , change as (
                select
                    symbol
                    , resolution
                    , time + interval '7 hours' as indexed_timestamp_
                    , close
                    , NVL(close - lag(close, 1) over (partition by symbol, resolution, time::date order by time), 0) as change
                from market_.index_ohlc
            )
        select
            max(change) as max_change
            , min(change) as min_change
            , extract(hour from indexed_timestamp_) as hour
            , extract(minute from indexed_timestamp_) as minute
        from change
        where
            symbol = 'VNINDEX'
            and resolution = 'MIN15'
            and indexed_timestamp_::date <= ((select indexed_timestamp_ from max_timestamp) - interval '1 days')::date
            and indexed_timestamp_::date >= ((select indexed_timestamp_ from max_timestamp) - interval '6 days')::date
        group by hour, minute
        order by hour, minute;
        """
    df = pd.read_sql_query(
        sql=sql,
        con=redshift_engine
    )

    return df


@st.cache_data
def get_symbol_effect_vni(url):
    try:
        input_data = requests.get(url).json()
        best_influence = sorted(input_data, key=lambda x: x.get("basketInfluence", float("-inf")), reverse=True)
        worst_influence = sorted(input_data, key=lambda x: x.get("basketInfluence", float("inf")))
        first_best = best_influence[0]['symbol']
        second_best = best_influence[1]['symbol']
        third_best = best_influence[2]['symbol']
        first_worst = worst_influence[0]['symbol']
        second_worst = worst_influence[1]['symbol']
        third_worst = worst_influence[2]['symbol']

        return first_best, second_best, third_best, first_worst, second_worst, third_worst
    except Exception as e:
        logging.warning("Error retrieve data from GET_SYMBOL_EFFECT_VNI_URL API")
        logging.warning(e)


@st.cache_data
def load_market_price_metric_range(start, end):
    start_date_string = datetime.strftime(start, '%Y-%m-%d')
    end_date_string = datetime.strftime(end, '%Y-%m-%d')
    sql = f"""
            select
                indexed_timestamp_
                , avg(pha_day_1m) as pha_day_1m
                , avg(vuot_dinh_1m) as vuot_dinh_1m
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
def load_net_buy_sell():
    sql = f"""
with
    sector_info as (
        select
            symbol_
            , nganhcap3
            , nganhcap2
            , nganhcap1
        from public.wifeed_bctc_thong_tin_co_ban_doanh_nghiep
        where indexed_timestamp_ in (select max(indexed_timestamp_) from wifeed_bctc_thong_tin_co_ban_doanh_nghiep)
    )
    , rank as (
        select
            *,
            row_number() over (partition by symbol order by indexed_timestamp_ desc ) as rank
        from public.symbol_all_metric
        where indexed_timestamp_::date = (select max(indexed_timestamp_) from public.symbol_all_metric)::date
    )
    , net_buy_sell_count as (
        select
            symbol
            , indexed_timestamp_::date as indexed_timestamp_
            , net_buy
            , net_sell
            , case
                when net_buy != 0 and net_sell = 0 then 1
                else 0
                end as net_buy_count
            , case
                when net_sell != 0 and net_buy = 0 then 1
                else 0
                end as net_sell_count
            , case
                when net_sell = 0 and net_buy = 0 then 1
                else 0
                end as neutral_count

        from rank
        where rank = 1
    )
select
    indexed_timestamp_
    , t1.nganhcap3
    , t1.nganhcap2
    , t1.nganhcap1
    , sum(t0.net_buy) as net_buy
    , sum(t0.net_sell) as net_sell
    , sum(t0.net_sell + t0.net_buy) as total_trading_value
    , sum(net_buy_count) as net_buy_count
    , sum(net_sell_count) as net_sell_count
    , sum(neutral_count) as neutral_count
    , avg(net_buy_count) as net_buy_ratio
    , avg(net_sell_count) as net_sell_ratio
    , avg(neutral_count) as neutral_ratio
    , count(*) as symbol_count
from
    net_buy_sell_count as t0
    join sector_info as t1
    on t0.symbol = t1.symbol_
group by
    t1.nganhcap3
    , t1.nganhcap2
    , t1.nganhcap1
    , t0.indexed_timestamp_
order by  indexed_timestamp_ desc;
        """
    df = pd.read_sql_query(
        sql=sql,
        con=dsai_engine
    )
    return df


@st.cache_data
def load_price_change_ratio():
    sql = f"""
with
    casted as (
         select
            indexed_timestamp_
            , name
            , CAST (price_increased_sum AS FLOAT) as price_increased_sum
            , CAST (price_decreased_sum AS FLOAT) as price_decreased_sum
            , CAST (price_unchanged_sum AS FLOAT) as price_unchanged_sum
            , CAST (no_price_info_sum AS FLOAT) as no_price_info_sum
        from public.price_metric_all_resolution
        where resolution = 'sector'
        and indexed_timestamp_ = (select max(indexed_timestamp_) from public.price_metric_all_resolution)
    )
    , symbol_count as (
        select
            *
            , price_increased_sum + price_decreased_sum + price_unchanged_sum + no_price_info_sum as symbol_count
        from casted
    )
select
    *
    , price_increased_sum / symbol_count as price_increased_mean
    , price_decreased_sum / symbol_count as price_decreased_mean
    , price_unchanged_sum / symbol_count as price_unchanged_mean
    , no_price_info_sum / symbol_count as no_price_info_mean
from symbol_count;

            """
    df = pd.read_sql_query(
        sql=sql,
        con=dsai_engine
    )
    return df


all_liquidity_same_period = liquidity_same_period()
latest_liquidity = load_lastest_liquidity()

latest_price_metric = load_latest_price_metric()
(
    market_price_increased_sum,
    market_price_decreased_sum,
    market_price_increased_mean,
    market_price_decreased_mean
) = latest_price_metric.query('name == "market"')[[
    'price_increased_sum',
    'price_decreased_sum',
    'price_increased_mean',
    'price_decreased_mean',
]].values.flatten().tolist()

(
    vni_new,
    vni_old,
    vni_change_value,
    vni_change_percent
) = get_current_vnindex(GET_CURRENT_VNINDEX_URL)

(
    first_best_influence,
    second_best_influence,
    third_best_influence,
    first_worst_influence,
    second_worst_influence,
    third_worst_influence
) = get_symbol_effect_vni(GET_SYMBOL_EFFECT_VNI_URL)

# Tổng quan VNINDEX


con0 = st.container()
c1, padding, c2 = con0.columns([1, 0.04, 1])


c1.header('VNINDEX')
gen_title_and_value_same_line(
    cell=c1,
    title='Điểm',
    value=gen_percentage_string(value=vni_change_percent * 100, prefix=f'{vni_new} (', suffix=')'),
    color=color_decider(vni_change_percent)
)

vni_liquidity = latest_liquidity.query('name == "HOSE"')['liquidity'].item()
gen_title_and_value_same_line(
    cell=c1,
    title='Giao dịch',
    value=gen_currency_string(value=float(vni_liquidity))
)

gen_title_and_value_same_line(
    cell=c1,
    title='Mã ảnh hưởng tích cực',
    value=', '.join([first_best_influence, second_best_influence, third_best_influence])
)

gen_title_and_value_same_line(
    cell=c1,
    title='Mã ảnh hưởng tiêu cực',
    value=', '.join([first_worst_influence, second_worst_influence, third_worst_influence])
)


c2.header('Mã tăng giảm giá')
gen_title_and_value_same_line(
    cell=c2,
    title='Số mã tăng',
    value=int(market_price_increased_sum),
    color=color_decider(market_price_increased_sum)
)

gen_title_and_value_same_line(
    cell=c2,
    title='Tỉ lệ mã tăng',
    value=gen_percentage_string(market_price_increased_mean),
    color=color_decider(market_price_increased_mean)
)

gen_title_and_value_same_line(
    cell=c2,
    title='Số mã giảm',
    value=int(market_price_decreased_sum),
    color=color_decider(market_price_decreased_sum, threshold=[1610])
)

gen_title_and_value_same_line(
    cell=c2,
    title='Tỉ lệ mã giảm',
    value=gen_percentage_string(market_price_decreased_mean),
    color=color_decider(market_price_decreased_mean, threshold=[1])
)

# Thanh khoản thị trường
con1 = st.container()
c1, _, c2 = con1.columns([1, 0.04, 1])


t1, t2 = c1.columns([1, 1])
t1.header('Tăng giảm VNINDEX')
vnindex_time_grain = t2.radio(
    "Thời gian",
    ["1 phút", "5 phút", '15 phút', "30 phút"],
    index=2,
    horizontal=True
)

if vnindex_time_grain == "1 phút":
    time_grain = 'MIN1'
elif vnindex_time_grain == "5 phút":
    time_grain = 'MIN5'
elif vnindex_time_grain == "15 phút":
    time_grain = 'MIN15'
elif vnindex_time_grain == "30 phút":
    time_grain = 'MIN30'
else:
    time_grain = 'MIN30'

vn_index_last_day = get_vnindex_latest_day(time_grain=time_grain)
vn_index_last_5_days = get_min_max_vnindex_point_last_5_days()
vn_index_last_day['time'] = vn_index_last_day.apply(lambda x: time(int(x['hour']), int(x['minute'])), axis=1)
vn_index_last_5_days['time'] = vn_index_last_5_days.apply(lambda x: time(int(x['hour']), int(x['minute'])), axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=vn_index_last_5_days["time"],
    y=vn_index_last_5_days["max_change"],
    fill='tozeroy',
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(211, 211, 211, 0.4)',
    line=dict(width=0),
    hovertemplate="<br>".join([
        "Thời gian: %{x}",
        "Thay đổi: %{y} điểm",
    ]),
    name="Mức tăng lớn nhất 5 ngày trước"
))
fig.add_trace(go.Scatter(
    x=vn_index_last_5_days["time"],
    y=vn_index_last_5_days["min_change"],
    fill='tozeroy',
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(211, 211, 211, 0.4)',
    line=dict(width=0),
    hovertemplate="<br>".join([
        "Thời gian: %{x}",
        "Thay đổi: %{y} điểm",
    ]),
    name="Mức tăng nhỏ nhất 5 ngày trước"
))
fig.add_trace(go.Scatter(
    x=vn_index_last_day["time"],
    y=vn_index_last_day["change"],
    customdata=vn_index_last_day["close"],
    line_shape="spline",
    mode='lines+markers+text',
    line=dict(width=3),
    marker=dict(size=8),
    text=vn_index_last_day["close"],
    textposition="bottom center",
    hovertemplate="<br>".join([
        "Thời gian: %{x}",
        "Điểm: %{customdata}",
        "Thay đổi: %{y} điểm",
    ]),
    name="Mức tăng VNINDEX"
))

fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=20,
    ),
    autosize=False,
    height=600,

    font=dict(
        size=16
    ),

    plot_bgcolor=BACKGROUND_COLOR,
    paper_bgcolor=BACKGROUND_COLOR,
)
fig = set_legend_at_top(fig)

fig.update_yaxes(
    range=[min(vn_index_last_5_days.min_change.min(), vn_index_last_day.change.min()) - 2,
           max(vn_index_last_5_days.max_change.max(), vn_index_last_day.change.max()) + 2]
)

c1.plotly_chart(fig, config=CONFIG, use_container_width=True)


# Thanh khoản thị trường
t1, t2 = c2.columns([1, 1])
t1.header('Thanh khoản thị trường')
sector_amount = t2.slider('Số lượng nhóm ngành', 0, 10, 5)
top_n_sector = latest_liquidity.query(
    'resolution == "market"'
    'or resolution == "sector"'
).sort_values(
    by='liquidity',
    ascending=False
)[['name']].values.flatten().tolist()[:sector_amount + 1]

# if not t2.checkbox('So sánh với thị trường', value=True):
#     top_n_sector.remove('Thị trường')

market_and_top_n_liquidity = all_liquidity_same_period[all_liquidity_same_period['name'].isin(top_n_sector)]

fig_liquidity = px.line(
    data_frame=market_and_top_n_liquidity,
    x="indexed_timestamp_",
    y="liquidity",
    color="name",
    markers=True,
    line_shape="spline",
    labels={"indexed_timestamp_": "", "liquidity": "", "name": "Tên"},
    height=600
)

fig_liquidity = line_chart_config(
    fig=fig_liquidity,
    marker_size=10,
    axis_title_size=16,
    tool_tip_font_size=25
).update_traces(
    selector={"name": "Thị trường"},
    line={"dash": "dash"}
)

c2.plotly_chart(fig_liquidity, config=CONFIG, use_container_width=True)

# Top ngành có số mã tăng giảm giá


con2 = st.container()
c1, _, c2 = con2.columns([1, 0.04, 1])

t1, t2 = c1.columns([1, 1])
t1.header('Top ngành có số mã tăng giá')
top_price_increase_amount = t2.slider('Số lượng ngành tăng', 0, 20, 15, step=5)
top_price_increase = latest_price_metric.query('name != "market"').sort_values(
    by='price_increased_sum',
    ascending=False
).head(top_price_increase_amount)

fig_price_increase = px.bar(
    data_frame=top_price_increase.sort_values(by='price_increased_sum'),
    x="price_increased_sum",
    y="name",
    orientation='h',
    labels={"price_increased_sum": "", "liquidity": "", "name": ""},
    color="price_increased_sum",
    color_continuous_scale=["rgb(127, 255, 127)", "green"],
    height=600
).update_traces(
    width=0.5,
    # marker_color='green',
).update_layout(
    xaxis=dict(
        side='top',
        tickfont=dict(size=16)
    ),
    yaxis=dict(
        side='right',
        tickfont=dict(size=16)
    ),
).update_coloraxes(showscale=False)

fig_price_increase = change_background_color(fig_price_increase)
c1.plotly_chart(fig_price_increase, config=CONFIG, use_container_width=True)

t1, t2 = c2.columns([1, 1])
t1.header('Top ngành có số mã giảm giá')
top_price_decrease_amount = t2.slider('Số lượng ngành giảm', 0, 20, 15, step=5)
top_price_decrease = latest_price_metric.query('name != "market"').sort_values(
    by='price_decreased_sum',
    ascending=False
).head(top_price_decrease_amount)

fig_price_decrease = px.bar(
    data_frame=top_price_decrease.sort_values(by='price_decreased_sum'),
    x="price_decreased_sum",
    y="name",
    orientation='h',
    labels={"price_decreased_sum": "", "liquidity": "", "name": ""},
    color="price_decreased_sum",
    color_continuous_scale=["rgb(255, 127, 127)", "red"],
    height=600
).update_traces(
    width=0.5,
    # marker_color='red',
).update_layout(
    xaxis=dict(
        side='top',
        tickfont=dict(size=16)
    ),
    yaxis=dict(
        side='right',
        tickfont=dict(size=16)
    ),
).update_coloraxes(showscale=False)

fig_price_decrease = change_background_color(fig_price_decrease)
c2.plotly_chart(fig_price_decrease, config=CONFIG, use_container_width=True)


con2 = st.container()
c1, _, c2 = con2.columns([1, 0.04, 1])

# Net buy net sell
t1, t2, t3 = c1.columns([1, 0.5, 0.5])
t1.header('Top ngành mua ròng')

net_buy_sell = load_net_buy_sell()
net_buy_ratio = t2.slider('% mua ròng', 0, 100, 10, step=5, format="%d%%")
net_buy_count = t3.slider('Số lượng mã mua ròng', 0, 20, 4, step=1)
top_net_buy = net_buy_sell.query(
    f'net_buy_ratio > {net_buy_ratio / 100} and net_buy_count > {net_buy_count}'
).sort_values(
    by='total_trading_value',
    ascending=False
)

fig_top_net_buy = px.scatter(
    data_frame=top_net_buy,
    x="net_buy",
    y="net_buy_ratio",
    color="nganhcap1",
    size="net_buy_count",
    # text="nganhcap3",
    labels={
        "nganhcap1": "Ngành cấp 1",
        "net_buy_count": "Số lượng mã mua ròng",
        "net_buy_ratio": "Tỉ lệ mã mua ròng",
        "net_buy": "Tổng khối lượng mua ròng"
    },
    height=600
).update_traces(
    marker=dict(
        line=dict(width=3, color='white')
    ),
    selector=dict(mode='markers')
)

fig_top_net_buy = set_legend_at_top(fig_top_net_buy)
fig_top_net_buy = change_background_color(fig_top_net_buy)
c1.plotly_chart(fig_top_net_buy, config=CONFIG, use_container_width=True)

# Net buy net sell
t1, t2, t3 = c2.columns([1, 0.5, 0.5])
t1.header('Top ngành bán ròng')

net_buy_sell = load_net_buy_sell()
net_sell_ratio = t2.slider('% bán ròng', 0, 100, 10, step=5, format="%d%%")
net_sell_count = t3.slider('Số lượng mã bán ròng', 0, 20, 4, step=1)
top_net_sell = net_buy_sell.query(
    f'net_sell_ratio > {net_sell_ratio / 100} and net_sell_count > {net_sell_count}'
).sort_values(
    by='total_trading_value',
    ascending=False
)

fig_top_net_sell = px.scatter(
    data_frame=top_net_sell,
    x="net_sell",
    y="net_sell_ratio",
    color="nganhcap1",
    size="net_sell_count",
    # text="nganhcap3",
    labels={
        "nganhcap1": "Ngành cấp 1",
        "net_sell_count": "Số lượng mã bán ròng",
        "net_sell_ratio": "Tỉ lệ mã bán ròng",
        "net_sell": "Tổng khối lượng bán ròng"
    },
    height=600
).update_traces(
    marker=dict(
        line=dict(width=3, color='white')
    ),
    selector=dict(mode='markers')
)

fig_top_net_sell = set_legend_at_top(fig_top_net_sell)
fig_top_net_sell = change_background_color(fig_top_net_sell)
c2.plotly_chart(fig_top_net_sell, config=CONFIG, use_container_width=True)


con3 = st.container()
c1, _, c2 = con3.columns([1, 0.04, 1])

# Net buy net sell
t1, t2 = c1.columns([1, 1])
t1.header('Tỉ lệ mã tăng giảm giá')

price_change_ratio = load_price_change_ratio()
# top_percentage_value = t2.slider('Top % khối lượng giao dịch', 0, 100, 80, step=5, format="%d%%")
min_symbol_count = t2.slider('Số lượng mã tối thiểu', 0, 20, 3, step=1, key='min_symbol_price_change_ratio')
price_change_ratio = price_change_ratio.query(
    f'symbol_count > {min_symbol_count}'
).sort_values(
    by='symbol_count',
    ascending=False
)
price_change_ratio = price_change_ratio.sort_values(by='price_increased_mean')

renamed_legend = {
    "value": "Tỉ lệ mã",
    "name": "Ngành cấp 3",
    "price_increased_mean": "Tăng giá",
    "price_unchanged_mean": "Giá giữ nguyên",
    "no_price_info_mean": "Không có dữ liệu",
    "price_decreased_mean": "Giảm giá"
}

fig_price_change_ratio = px.bar(
    data_frame=price_change_ratio,
    x=["price_increased_mean", "price_unchanged_mean", "no_price_info_mean",  "price_decreased_mean"],
    y="name",
    orientation='h',
    # text="nganhcap3",
    color_discrete_map={
        "price_increased_mean": "#41dc8e",
        "price_unchanged_mean": "#daa520",
        "no_price_info_mean": "#d3d3d3",
        "price_decreased_mean": "#EF3A4C",
    },
    labels=renamed_legend,
    height=800
).update_traces(
    marker=dict(
        line=dict(width=3, color='DarkSlateGrey')
    ),
    selector=dict(mode='markers')
).update_layout(
    yaxis=dict(
        side='right',
        tickfont=dict(size=16)
    ),
    legend_title=None
).for_each_trace(lambda t: t.update(name=renamed_legend[t.name]))

fig_price_change_ratio = axis_percentage(fig_price_change_ratio, axis='x')
fig_price_change_ratio = set_axis_title_size(fig_price_change_ratio, size=16)
fig_price_change_ratio = set_legend_at_top(fig_price_change_ratio)
fig_price_change_ratio = change_background_color(fig_price_change_ratio)

c1.plotly_chart(fig_price_change_ratio, config=CONFIG, use_container_width=True)


# Net buy net sell
t1, t2, t3 = c2.columns([1, 0.5, 0.5])
t1.header('Tỉ lệ mã mua bán ròng')

net_buy_sell = load_net_buy_sell()
top_percentage_value = t2.slider('Top % khối lượng giao dịch', 0, 100, 80, step=5, format="%d%%")
min_symbol_count = t3.slider('Số lượng mã tối thiểu', 0, 20, 3, step=1, key='min_symbol_net_buy_sell')
net_buy_sell = net_buy_sell.query(
    f'symbol_count > {min_symbol_count}'
)
net_buy_sell = net_buy_sell.sort_values(
    by='symbol_count',
    ascending=False
).head(int(len(net_buy_sell) * top_percentage_value/100))


net_buy_sell = net_buy_sell.sort_values(by='net_buy_ratio')

renamed_legend = {
    "value": "Tỉ lệ mã",
    "nganhcap3": "Ngành cấp 3",
    "net_sell_ratio": "Bán ròng",
    "neutral_ratio": "Trung tính",
    "net_buy_ratio": "Mua ròng"
}
fig_top_net_sell = px.bar(
    data_frame=net_buy_sell,
    x=["net_buy_ratio", "neutral_ratio", "net_sell_ratio"],
    y="nganhcap3",
    orientation='h',
    color_discrete_map={
        "net_buy_ratio": "#41dc8e",
        "neutral_ratio": "#d3d3d3",
        "net_sell_ratio": "#EF3A4C",
    },
    labels=renamed_legend,
    height=800
).update_traces(
    marker=dict(
        line=dict(width=3, color='DarkSlateGrey')
    ),
    selector=dict(mode='markers')
).update_layout(
    yaxis=dict(
        side='right',
        tickfont=dict(size=16)
    ),
    legend_title=None
).for_each_trace(lambda t: t.update(name=renamed_legend[t.name]))

fig_top_net_sell = axis_percentage(fig_top_net_sell, axis='x')
fig_top_net_sell = set_axis_title_size(fig_top_net_sell, size=16)
fig_top_net_sell = set_legend_at_top(fig_top_net_sell)
fig_top_net_sell = change_background_color(fig_top_net_sell)

c2.plotly_chart(fig_top_net_sell, config=CONFIG, use_container_width=True)


con4 = st.container()
c1, _, c2 = con4.columns([1, 0.04, 1])
c1.header('Thanh khoản so với quá khứ')
t1, t2 = c1.columns([1, 1])
past_value = t1.radio(
    "Thời gian",
    ["Hôm qua", "Trung bình 7 ngày"],
    index=1,
    horizontal=True
)

compare_col = 'liquidity_vs_1d_ago' if past_value == 'Hôm qua' else 'liquidity_vs_avg_7d_ago'
top_sector_vs_avg_7d_ago = t2.slider('Top % khối lượng giao dịch', 0, 100, 80, step=5, format="%d%%", key='top_sector_vs_avg_7d_ago')
lastest_liquidity = load_lastest_liquidity()

lastest_liquidity = lastest_liquidity.fillna(0).query('resolution == "sector"').sort_values(
    by='liquidity',
    ascending=False
).head(int(len(lastest_liquidity) * top_sector_vs_avg_7d_ago/100))

lastest_liquidity = lastest_liquidity.sort_values(by=compare_col)
lastest_liquidity["color"] = np.where(lastest_liquidity[compare_col] < 0, 'red', 'green')

fig_liquidity_vs_1d_ago = px.bar(
    data_frame=lastest_liquidity,
    x=compare_col,
    y="name",
    orientation='h',
    labels={compare_col: "", "liquidity": "", "name": ""},
    height=600
).update_traces(
    width=0.5,

    marker_color=lastest_liquidity["color"],
).update_layout(
    xaxis=dict(
        side='top',
        tickfont=dict(size=16)
    ),
    yaxis=dict(
        side='right',
        tickfont=dict(size=16)
    ),
).update_coloraxes(
    showscale=False
).update_xaxes(
    tickformat=',.0%',
)


fig_liquidity_vs_1d_ago = change_background_color(fig_liquidity_vs_1d_ago)
c1.plotly_chart(fig_liquidity_vs_1d_ago, config=CONFIG, use_container_width=True)


# df = lastest_liquidity.sort_values(by=compare_col)
# fig = go.Figure(data=[
#     go.Bar(
#         x=df[compare_col],
#         y=df['name'],
#         marker=dict(
#             color=df[compare_col],
#             colorscale=[[0, 'red'],
#                         [1, 'green']],
#             showscale=True,
#             colorbar=dict(title="value"),
#         ),
#     )
# ])
#
# c2.plotly_chart(fig, config=CONFIG, use_container_width=True)


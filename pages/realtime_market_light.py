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
st.set_page_config(page_title="Market Overview", page_icon="📈", initial_sidebar_state="collapsed", layout="wide")

# Styling
with open('css/realtime_market_light.css') as f:
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


def get_min_max_vnindex_point_last_5_days(time_grain):
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
            and resolution = '{time_grain}'
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


@st.cache_data
def get_liquidity_and_min_max_5_day(timestamp: datetime):
    sql = f"""
        select
            *
        from public.liquidity_all_resolution
        where indexed_timestamp_ > '{timestamp.strftime('%Y-%m-%d')}'::date - interval '5 days'
        and indexed_timestamp_::date <= '{timestamp.strftime('%Y-%m-%d')}'::date
        order by indexed_timestamp_;
        """
    df = pd.read_sql_query(
        sql=sql,
        con=dsai_engine
    )

    df["indexed_timestamp_"] = pd.to_datetime(df["indexed_timestamp_"])
    df = df[df['resolution'] == 'market']
    df['minute'] = df['indexed_timestamp_'].apply(lambda x: x.minute)
    df['time'] = df['indexed_timestamp_'].dt.time

    # df = df[df['minute'].isin([0, 15, 30, 45])]
    df = df[df['minute'].isin([0, 30])]
    df = df.sort_values(by='indexed_timestamp_', ascending=False)

    today = df[df['indexed_timestamp_'].dt.date == timestamp.date()]

    five_day_ago = df[df['indexed_timestamp_'].dt.date < timestamp.date()]
    five_day_ago = five_day_ago.groupby('time').agg(
        max_liquidity=('liquidity', 'max'),
        min_liquidity=('liquidity', 'min')
    )

    five_day_ago.reset_index(inplace=True)
    today.reset_index(inplace=True)
    return today, five_day_ago


def get_top_n_sector_liquidity(timestamp: datetime, n=8):
    sql = f"""
        select
            *
        from public.liquidity_all_resolution
        where indexed_timestamp_::date = '{timestamp.strftime('%Y-%m-%d')}'::date
        order by indexed_timestamp_;
        """
    df = pd.read_sql_query(
        sql=sql,
        con=dsai_engine
    )

    df["indexed_timestamp_"] = pd.to_datetime(df["indexed_timestamp_"])
    df = df[df['resolution'] == 'sector']
    df['minute'] = df['indexed_timestamp_'].apply(lambda x: x.minute)
    df['time'] = df['indexed_timestamp_'].dt.time

    # df = df[df['minute'].isin([0, 15, 30, 45])]
    df = df[df['minute'].isin([0, 30])]

    top_8_liquidity = df[df['indexed_timestamp_'] == df['indexed_timestamp_'].max()].sort_values(by='liquidity', ascending=False).head(n)['name'].unique().tolist()
    today = df[df['indexed_timestamp_'].dt.date == timestamp.date()]
    today = today[today['name'].isin(top_8_liquidity)]

    today['ratio'] = today['liquidity'] / today.groupby('time')['liquidity'].transform(pd.Series.sum)
    today = today.sort_values(by='indexed_timestamp_')
    return today


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
con0.markdown(
    '''
    <p style="font-size: 30px; margin: 0 0 0 -15px;">
     <b>Báo cáo thị trường thời gian thực</b>
     </p>
    ''',
    unsafe_allow_html=True
)
c1, _, c2 = con0.columns([1, 0.04, 1])

t1, _ = c1.columns([1, 1])
t1.header('VNINDEX')

t1, t2, t3, t4 = c1.columns([1, 1, 1, 1])


gen_big_number(
    cell=t1,
    title='Điểm',
    value=gen_percentage_string(value=vni_change_percent * 100, prefix=f'{vni_new} (', suffix=')'),
    color=color_decider(vni_change_percent)
)

vni_liquidity = latest_liquidity.query('name == "HOSE"')['liquidity'].item()
gen_big_number(
    cell=t2,
    title='Giao dịch',
    value=gen_currency_string(value=float(vni_liquidity))
)

gen_big_number(
    cell=t3,
    title='Mã ảnh hưởng tích cực',
    value=', '.join([first_best_influence, second_best_influence, third_best_influence])
)

gen_big_number(
    cell=t4,
    title='Mã ảnh hưởng tiêu cực',
    value=', '.join([first_worst_influence, second_worst_influence, third_worst_influence])
)


t1, _ = c2.columns([1, 1])
t1.header('Mã tăng giảm giá')
t5, t6, t7, t8 = c2.columns([1, 1, 1, 1])

gen_big_number(
    cell=t5,
    title='Số mã tăng giá',
    value=int(market_price_increased_sum),
    color=color_decider(market_price_increased_sum)
)

gen_big_number(
    cell=t6,
    title='Tỉ lệ mã tăng giá',
    value=gen_percentage_string(market_price_increased_mean),
    color=color_decider(market_price_increased_mean)
)

gen_big_number(
    cell=t7,
    title='Số mã tăng giá',
    value=int(market_price_decreased_sum),
    color=color_decider(market_price_decreased_sum, threshold=[1610])
)

gen_big_number(
    cell=t8,
    title='Tỉ lệ mã tăng giá',
    value=gen_percentage_string(market_price_decreased_mean),
    color=color_decider(market_price_decreased_mean, threshold=[1])
)


# con0 = st.container()
# c1, _, c2 = con0.columns([1, 0.04, 1])
# c1.markdown(f'''
# - **Ý nghĩa**: Biểu diễn mức thay đổi điểm VNINDEX so với 5 ngày trước đó
# - **Thời gian**: Khoảng thời gian giữa các thay đổi
# ''', unsafe_allow_html=True)
#
#
# c2.markdown(f'''
# - **Ý nghĩa**: Lịch sử thanh khoản của top N nhóm ngành có thanh khoản lớn nhất
# - **Số lượng nhóm ngành**: Chọn số lượng nhóm ngành để biểu diễn
# ''', unsafe_allow_html=True)

# Thanh khoản thị trường
# Thanh khoản thị trường
con1 = st.container()
c1, _, c2 = con1.columns([1, 0.04, 1])

t1, t2 = c1.columns([1, 1.2])
t1.header('Tăng giảm VNINDEX')
vnindex_time_grain = t2.radio(
    "Thời gian",
    ["5 phút", '15 phút', "30 phút"],
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
vn_index_last_5_days = get_min_max_vnindex_point_last_5_days(time_grain=time_grain)

vn_index_last_day['time'] = vn_index_last_day.apply(lambda x: time(int(x['hour']), int(x['minute'])).strftime("%H:%M"),
                                                    axis=1)
vn_index_last_5_days['time'] = vn_index_last_5_days.apply(
    lambda x: time(int(x['hour']), int(x['minute'])).strftime("%H:%M"), axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=vn_index_last_5_days["time"],
    y=vn_index_last_5_days["max_change"],
    fill=None,
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(211, 211, 211, 0.7)',
    line=dict(width=0),
    hovertemplate="<br>".join([
        "%{y} điểm",
    ]),
    name="Mức tăng lớn nhất 5 ngày trước"
))
fig.add_trace(go.Scatter(
    x=vn_index_last_5_days["time"],
    y=vn_index_last_5_days["min_change"],
    fill='tonexty',
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(211, 211, 211, 0.7)',
    line=dict(width=0),
    hovertemplate="<br>".join([
        "%{y} điểm",
    ]),
    name="Mức tăng nhỏ nhất 5 ngày trước"
))
fig.add_trace(go.Scatter(
    x=vn_index_last_day["time"],
    y=vn_index_last_day["change"],
    customdata=vn_index_last_day["close"],
    line_shape="spline",
    mode='lines+markers+text',
    line=dict(width=3, color='rgb(0, 204, 150)'),

    marker=dict(size=8),
    text=vn_index_last_day["close"],
    textposition="top center",
    hovertemplate="<br>".join([
        "%{customdata} điểm",
        "Thay đổi VNINDEX: %{y} điểm",
    ]),
    name="VNINDEX"
))

fig.update_layout(
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="white",
        font_size=20,
    ),
    autosize=False,
    height=700,

    font=dict(
        size=16
    ),
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),

    plot_bgcolor=BACKGROUND_COLOR,
    paper_bgcolor=BACKGROUND_COLOR,
)
fig = set_legend_at_top(fig)
fig.update_xaxes(
    tickfont=dict(color=LIGHT_BLACK),
).update_yaxes(
    tickfont=dict(color=LIGHT_BLACK),
    range=[min(vn_index_last_5_days.min_change.min(), vn_index_last_day.change.min()) - 2,
           max(vn_index_last_5_days.max_change.max(), vn_index_last_day.change.max()) + 2]
)

fig = set_axis_title_size(fig)

c1.plotly_chart(fig, config=CONFIG, use_container_width=True)

# Thanh khoản thị trường
t1, t2 = c2.columns([1, 1])
t1.header('Thanh khoản thị trường')

market_liquidity_today, market_liquidity_5d_ago = get_liquidity_and_min_max_5_day(datetime(2023, 10, 12))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=market_liquidity_5d_ago["time"],
    y=market_liquidity_5d_ago["max_liquidity"],
    # fill=None,
    fill='tozeroy',
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(211, 211, 211, 0.7)',
    line=dict(width=0),
    hovertemplate="<br>".join([
        "%{y} nghìn tỉ",
    ]),
    name="Thanh khoản lớn nhất 5 ngày trước"
))
fig.add_trace(go.Scatter(
    x=market_liquidity_5d_ago["time"],
    y=market_liquidity_5d_ago["min_liquidity"],
    # fill='tonexty',
    fill='tozeroy',
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(211, 211, 211, 0.7)',
    line=dict(width=0),
    hovertemplate="<br>".join([
        "%{y} nghìn tỉ",
    ]),
    name="Thanh khoản nhỏ nhất 5 ngày trước"
))
fig.add_trace(go.Scatter(
    x=market_liquidity_today["time"],
    y=market_liquidity_today["liquidity"],
    line_shape="spline",
    mode='lines+markers+text',
    line=dict(width=3, color=DNSE_RED),
    marker=dict(size=8),
    text=market_liquidity_today["liquidity"].astype('int'),
    textposition="top left",
    hovertemplate="<br>".join([
        "Thanh khoản hiện tại: %{y} nghìn tỉ",
    ]),
    name="VNINDEX"
))

fig.update_layout(
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="white",
        font_size=20,
    ),
    autosize=False,
    height=300,

    font=dict(
        size=16
    ),
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),

    plot_bgcolor=BACKGROUND_COLOR,
    paper_bgcolor=BACKGROUND_COLOR,
)
fig = set_legend_at_top(fig)
fig.update_xaxes(
    tickfont=dict(color=LIGHT_BLACK),
).update_yaxes(
    tickfont=dict(color=LIGHT_BLACK),
    range=[market_liquidity_5d_ago.min_liquidity.min() - 2,
           market_liquidity_5d_ago.max_liquidity.max() + 2]
)

fig = set_axis_title_size(fig)

c2.plotly_chart(fig, config=CONFIG, use_container_width=True)

con0 = st.container()

c2.header(f'Tỉ lệ thanh khoản theo thời gian')

top_n_sector = get_top_n_sector_liquidity(datetime(2023, 10, 16))

fig_liquidity = px.area(
    data_frame=top_n_sector,
    x=top_n_sector["time"],
    y="ratio",
    color="name",
    markers=True,
    labels={"time": "Thời gian", "ratio": "Tỉ lệ thanh khoản", "name": "Ngành"},
    height=600
)

# fig_liquidity.add_scatter(x=[fig.data[0].x[-1]], y=[fig.data[0].y[-1]])

fig_liquidity.update_traces(
    hovertemplate='<br>Tỉ lệ thanh khoản: %{y}',
)

config = {'displayModeBar': False}

fig_liquidity = line_chart_config(
    fig=fig_liquidity,
    marker_size=8,
    axis_title_size=16,
    tool_tip_font_size=18,
    y_axis_data_type='percentage',
    legend_position='top'
)

fig_liquidity.update_layout(
    hovermode="x unified",
    yaxis=dict(hoverformat=',.0%'),
    font_color=BLACK,
    height=350,
    legend=dict(
        orientation="v",
        yanchor="auto",
        y=1,
        xanchor="right",  # changed
        x=-0.3
    )
).update_xaxes(
    tickfont=dict(color=LIGHT_BLACK),
    range=[-0.05, 8.05]
).update_yaxes(
    tickfont=dict(color=LIGHT_BLACK),
    range=[0, 1.05]
)

c2.plotly_chart(fig_liquidity, config=config, use_container_width=True)

# Thanh khoản thị trường
# t1, t2 = c2.columns([1, 1])
# t1.header('Thanh khoản thị trường')
# sector_amount = t2.slider('Số lượng nhóm ngành', 0, 10, 5)
# top_n_sector = latest_liquidity.query(
#     'resolution == "market"'
#     'or resolution == "sector"'
# ).sort_values(
#     by='liquidity',
#     ascending=False
# )[['name']].values.flatten().tolist()[:sector_amount + 1]
#
# # if not t2.checkbox('So sánh với thị trường', value=True):
# #     top_n_sector.remove('Thị trường')
#
# market_and_top_n_liquidity = all_liquidity_same_period[all_liquidity_same_period['name'].isin(top_n_sector)]
#
# fig_liquidity = px.line(
#     data_frame=market_and_top_n_liquidity,
#     x="indexed_timestamp_",
#     y="liquidity",
#     color="name",
#     markers=True,
#     line_shape="spline",
#     labels={"indexed_timestamp_": "Thời gian", "liquidity": "Thanh khoản", "name": "Ngành"},
#     height=600
# )
#
# fig_liquidity.update_traces(
#     hovertemplate='<br>Thanh khoản: %{y}',
# )
#
# fig_liquidity = line_chart_config(
#     fig=fig_liquidity,
#     marker_size=10,
#     axis_title_size=16,
#     tool_tip_font_size=18
# ).update_traces(
#     selector={"name": "Thị trường"},
#     line={"dash": "dash"}
# )
#
# fig_liquidity.update_layout(
#     hovermode="x unified",
#     margin=dict(
#         t=10,
#         b=20,
#         pad=5
#     ),
# )
#
# c2.plotly_chart(fig_liquidity, config=CONFIG, use_container_width=True)


# Net buy net sell
con3 = st.container()
c1, _, c2 = con3.columns([1, 0.04, 1])

t1, t2 = c1.columns([1, 1])
t1.header('Tỉ lệ mã tăng giảm giá')

price_change_ratio = load_price_change_ratio()
# top_percentage_value = t2.slider('Top % giao dịch', 0, 100, 80, step=5, format="%d%%")
min_symbol_count = t2.slider('Số lượng mã tối thiểu', 0, 20, 5, step=1, key='min_symbol_price_change_ratio')
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
        "price_increased_mean": CYAN,
        "price_unchanged_mean": AMBER,
        "no_price_info_mean": LIGHT_GRAY,
        "price_decreased_mean": LIGHT_RED,
    },
    labels=renamed_legend,
    height=800
).update_traces(
    marker=dict(
        line=dict(width=3, color='DarkSlateGrey')
    ),
    selector=dict(mode='markers'),
    hovertemplate='<br>%{y}',
).update_layout(
    hovermode="y unified",
    hoverlabel=dict(font=dict(size=18)),
    yaxis_title=None,
    yaxis=dict(
        side='right',
        tickfont=dict(size=14)
    ),
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
    legend_title=None
).for_each_trace(
    lambda t: t.update(name=renamed_legend[t.name])
).update_xaxes(
    tickfont=dict(color=LIGHT_BLACK),
).update_yaxes(
    tickfont=dict(color=LIGHT_BLACK),
)

fig_price_change_ratio = axis_percentage(fig_price_change_ratio, axis='x')
fig_price_change_ratio = set_axis_title_size(fig_price_change_ratio, size=14)
fig_price_change_ratio = set_legend_at_top(fig_price_change_ratio)
fig_price_change_ratio = change_background_color(fig_price_change_ratio)

c1.plotly_chart(fig_price_change_ratio, config=CONFIG, use_container_width=True)


# Net buy net sell
t1, t2, t3 = c2.columns([1, 0.5, 0.5])
t1.header('Tỉ lệ mã mua bán ròng')

net_buy_sell = load_net_buy_sell()
top_percentage_value = t2.slider('Top % giao dịch', 0, 100, 80, step=5, format="%d%%")
min_symbol_count = t3.slider('Số lượng mã tối thiểu', 0, 20, 5, step=1, key='min_symbol_net_buy_sell')
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
        "net_buy_ratio": CYAN,
        "neutral_ratio": LIGHT_GRAY,
        "net_sell_ratio": LIGHT_RED,
    },
    labels=renamed_legend,
    height=800
).update_traces(
    hovertemplate='<br>%{y}',
    marker=dict(
        line=dict(width=3, color='DarkSlateGrey')
    ),
    selector=dict(mode='markers')
).update_layout(
    yaxis_title=None,
    hovermode="y unified",
    hoverlabel=dict(font=dict(size=18)),
    yaxis=dict(
        side='right',
        tickfont=dict(size=14)
    ),
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
    legend_title=None
).for_each_trace(
    lambda t: t.update(name=renamed_legend[t.name])
).update_xaxes(
    tickfont=dict(color=LIGHT_BLACK),
).update_yaxes(
    tickfont=dict(color=LIGHT_BLACK),
)

fig_top_net_sell = axis_percentage(fig_top_net_sell, axis='x')
fig_top_net_sell = set_axis_title_size(fig_top_net_sell, size=14)
fig_top_net_sell = set_legend_at_top(fig_top_net_sell)
fig_top_net_sell = change_background_color(fig_top_net_sell)

c2.plotly_chart(fig_top_net_sell, config=CONFIG, use_container_width=True)


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
    color="nganhcap3",
    size="net_buy_count",
    # text="nganhcap3",
    labels={
        "nganhcap3": "Ngành cấp 3",
        "net_buy_count": "Số lượng mã mua ròng",
        "net_buy_ratio": "Tỉ lệ mã mua ròng",
        "net_buy": "Tổng khối lượng mua ròng"
    },
    height=600
).update_traces(
    marker=dict(
        line=dict(width=3, color='DarkSlateGrey')
    ),
    selector=dict(mode='markers')
).update_xaxes(
    tickfont=dict(color=LIGHT_BLACK),
).update_yaxes(
    tickformat=',.0%',
    tickfont=dict(color=LIGHT_BLACK),
)

fig_top_net_buy = set_legend_at_top(fig_top_net_buy)
fig_top_net_buy = change_background_color(fig_top_net_buy)
fig_top_net_buy = set_axis_title_size(fig_top_net_buy)
fig_top_net_buy = set_tool_tip_font_size(fig=fig_top_net_buy, size=18)

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
    color="nganhcap3",
    size="net_sell_count",
    # text="nganhcap3",
    labels={
        "nganhcap3": "Ngành cấp 3",
        "net_sell_count": "Số lượng mã bán ròng",
        "net_sell_ratio": "Tỉ lệ mã bán ròng",
        "net_sell": "Tổng khối lượng bán ròng"
    },
    height=600
).update_traces(
    marker=dict(
        line=dict(width=3, color='DarkSlateGrey')
    ),
    selector=dict(mode='markers')
).update_xaxes(
    tickfont=dict(color=LIGHT_BLACK),
).update_yaxes(
    tickformat=',.0%',
    title_font_color=LIGHT_BLACK,
    tickfont=dict(color=LIGHT_BLACK),
)


fig_top_net_sell = set_legend_at_top(fig_top_net_sell)
fig_top_net_sell = change_background_color(fig_top_net_sell)
fig_top_net_sell = set_axis_title_size(fig_top_net_sell)
fig_top_net_sell = set_tool_tip_font_size(fig=fig_top_net_sell, size=18)
c2.plotly_chart(fig_top_net_sell, config=CONFIG, use_container_width=True)

# So sánh với quá khứ
con4 = st.container()
c1, _, c2 = con4.columns([1, 0.04, 1])
t1, t2 = c1.columns([1, 1])
t1.header('Thanh khoản so với hôm qua')

top_sector_vs_1d_ago = t2.slider('Top % giao dịch', 0, 100, 60, step=5, format="%d%%", key='top_sector_vs_1d_ago')
lastest_liquidity = load_lastest_liquidity()

lastest_liquidity = lastest_liquidity.fillna(0).query('resolution == "sector"').sort_values(
    by='liquidity',
    ascending=False
).head(int(len(lastest_liquidity) * top_sector_vs_1d_ago/100))

lastest_liquidity = lastest_liquidity.sort_values(by='liquidity_vs_1d_ago')
lastest_liquidity["color"] = np.where(lastest_liquidity['liquidity_vs_1d_ago'] < 0, LIGHT_RED, CYAN)

fig_liquidity_vs_1d_ago = px.bar(
    data_frame=lastest_liquidity,
    x='liquidity_vs_1d_ago',
    y="name",
    orientation='h',
    labels={'liquidity_vs_1d_ago': "", "liquidity": "", "name": ""},
    height=800
).update_traces(
    width=0.5,
    hovertemplate='<br>%{y}: %{x}',
    marker_color=lastest_liquidity["color"],
).update_layout(
    xaxis=dict(
        side='top',
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        side='right',
        tickfont=dict(size=12)
    ),
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
).update_coloraxes(
    showscale=False
).update_xaxes(
    tickformat=',.0%',
    tickfont=dict(size=12, color=LIGHT_BLACK),
).update_yaxes(
    tickfont=dict(color=LIGHT_BLACK),
)


fig_liquidity_vs_1d_ago = set_tool_tip_font_size(fig=fig_liquidity_vs_1d_ago, size=18)
fig_liquidity_vs_1d_ago = change_background_color(fig_liquidity_vs_1d_ago)
c1.plotly_chart(fig_liquidity_vs_1d_ago, config=CONFIG, use_container_width=True)


t1, t2 = c2.columns([1, 1])
t1.header('Thanh khoản so với trung bình 7 ngày')

top_sector_vs_avg_7d_ago = t2.slider('Top % giao dịch', 0, 100, 60, step=5, format="%d%%", key='top_sector_vs_avg_7d_ago')
lastest_liquidity = load_lastest_liquidity()

lastest_liquidity = lastest_liquidity.fillna(0).query('resolution == "sector"').sort_values(
    by='liquidity',
    ascending=False
).head(int(len(lastest_liquidity) * top_sector_vs_avg_7d_ago/100))

lastest_liquidity = lastest_liquidity.sort_values(by='liquidity_vs_avg_7d_ago')
lastest_liquidity["color"] = np.where(lastest_liquidity['liquidity_vs_avg_7d_ago'] < 0, LIGHT_RED, CYAN)

fig_liquidity_vs_avg_7d_ago = px.bar(
    data_frame=lastest_liquidity,
    x='liquidity_vs_avg_7d_ago',
    y="name",
    orientation='h',
    labels={'liquidity_vs_avg_7d_ago': "", "liquidity": "", "name": ""},
    height=800
).update_traces(
    width=0.5,
    hovertemplate='<br>%{y}: %{x}',
    marker_color=lastest_liquidity["color"],
).update_layout(
    xaxis=dict(
        side='top',
        tickfont=dict(size=12, color=LIGHT_BLACK)
    ),
    yaxis=dict(
        side='right',
        tickfont=dict(size=12, color=LIGHT_BLACK)
    ),
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
).update_coloraxes(
    showscale=False
).update_xaxes(
    tickformat=',.0%',
)

fig_liquidity_vs_avg_7d_ago = set_tool_tip_font_size(fig=fig_liquidity_vs_avg_7d_ago, size=18)
fig_liquidity_vs_avg_7d_ago = change_background_color(fig_liquidity_vs_avg_7d_ago)
c2.plotly_chart(fig_liquidity_vs_avg_7d_ago, config=CONFIG, use_container_width=True)


# Top ngành có số mã tăng giá giảm giá
con2 = st.container()
c1, _, c2 = con2.columns([1, 0.04, 1])

t1, t2 = c1.columns([1, 1])
t1.header('Top ngành có số mã tăng giá')
top_price_increase_amount = t2.slider('Số lượng ngành', 0, 20, 15, step=5, key='top_price_increase_amount')
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
    color_continuous_scale=[LIGHT_CYAN, CYAN],
    height=600
).update_traces(
    width=0.5,
    # marker_color=CYAN,
).update_layout(
    xaxis=dict(
        side='top',
        tickfont=dict(size=16, color=LIGHT_BLACK)
    ),
    yaxis=dict(
        side='right',
        tickfont=dict(size=16, color=LIGHT_BLACK)
    ),
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
).update_coloraxes(showscale=False)

fig_price_increase = change_background_color(fig_price_increase)
c1.plotly_chart(fig_price_increase, config=CONFIG, use_container_width=True)

t1, t2 = c2.columns([1, 1])
t1.header('Top ngành có số mã giảm giá')
top_price_decrease_amount = t2.slider('Số lượng ngành', 0, 20, 15, step=5, key='top_price_decrease_amount')
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
    color_continuous_scale=[SUPER_LIGHT_RED, LIGHT_RED],
    height=600
).update_traces(
    width=0.5,
    # marker_color=LIGHT_RED,
).update_layout(
    xaxis=dict(
        side='top',
        tickfont=dict(size=16, color=LIGHT_BLACK),
    ),
    yaxis=dict(
        side='right',
        tickfont=dict(size=16, color=LIGHT_BLACK)
    ),
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
).update_coloraxes(showscale=False)

fig_price_decrease = change_background_color(fig_price_decrease)
c2.plotly_chart(fig_price_decrease, config=CONFIG, use_container_width=True)
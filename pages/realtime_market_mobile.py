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
with open('css/realtime_market_mobile.css') as f:
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


all_liquidity_same_period = liquidity_same_period()
latest_liquidity = load_lastest_liquidity()

latest_price_metric = load_latest_price_metric()
(
    market_price_increased_sum,
    market_price_increased_mean,
    market_price_decreased_sum,
    market_price_decreased_mean
) = latest_price_metric.query('name == "market"')[[
    'price_increased_sum',
    'price_increased_mean',
    'price_decreased_sum',
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

st.markdown(
    '''
    <p style="font-size: 25px; margin: 0 0 0 0;">
     <b>Báo cáo thị trường thời gian thực</b>
     </p>
    ''',
    unsafe_allow_html=True
)

st.header('VNINDEX')
gen_title_and_value_same_line(
    cell=st,
    title='Điểm',
    value=gen_percentage_string(value=vni_change_percent * 100, prefix=f'{vni_new} (', suffix=')'),
    color=color_decider(vni_change_percent)
)

vni_liquidity = latest_liquidity.query('name == "HOSE"')['liquidity'].item()
gen_title_and_value_same_line(
    cell=st,
    title='Giao dịch',
    value=gen_currency_string(value=float(vni_liquidity))
)

gen_title_and_value_same_line(
    cell=st,
    title='Mã ảnh hưởng tích cực',
    value=', '.join([first_best_influence, second_best_influence, third_best_influence])
)

gen_title_and_value_same_line(
    cell=st,
    title='Mã ảnh hưởng tiêu cực',
    value=', '.join([first_worst_influence, second_worst_influence, third_worst_influence])
)

st.header('Mã tăng giảm giá')
gen_title_and_value_same_line(
    cell=st,
    title='Số mã tăng giá',
    value=int(market_price_decreased_sum),
    color=color_decider(market_price_decreased_sum)
)

gen_title_and_value_same_line(
    cell=st,
    title='Tỉ lệ mã tăng giá',
    value=gen_percentage_string(market_price_decreased_sum),
    color=color_decider(market_price_decreased_sum)
)

gen_title_and_value_same_line(
    cell=st,
    title='Số mã tăng giá',
    value=int(market_price_decreased_sum),
    color=color_decider(market_price_decreased_sum, threshold=[1610])
)

gen_title_and_value_same_line(
    cell=st,
    title='Tỉ lệ mã tăng giá',
    value=gen_percentage_string(market_price_decreased_mean),
    color=color_decider(market_price_decreased_mean, threshold=[1])
)

# con0 = st.container()
# st, _, st = con0.columns([1, 0.04, 1])
# st.markdown(f'''
# - **Ý nghĩa**: Biểu diễn mức thay đổi điểm VNINDEX so với 5 ngày trước đó
# - **Thời gian**: Khoảng thời gian giữa các thay đổi
# ''', unsafe_allow_html=True)
#
#
# st.markdown(f'''
# - **Ý nghĩa**: Lịch sử thanh khoản của top N nhóm ngành có thanh khoản lớn nhất
# - **Số lượng nhóm ngành**: Chọn số lượng nhóm ngành để biểu diễn
# ''', unsafe_allow_html=True)

# Thanh khoản thị trường
st.header('Tăng giảm VNINDEX')
vnindex_time_grain = st.radio(
    "Thời gian",
    ["5 phút", '15 phút', "30 phút"],
    index=1,
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
    fill='tozeroy',
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(211, 211, 211, 0.7)',
    line=dict(width=0),
    hovertemplate="<br>".join([
        "%{y} điểm",
    ]),
    name="Mức tăng lớn nhất"
))
fig.add_trace(go.Scatter(
    x=vn_index_last_5_days["time"],
    y=vn_index_last_5_days["min_change"],
    fill='tozeroy',
    line_shape="spline",
    mode='lines',
    fillcolor='rgba(211, 211, 211, 0.7)',
    line=dict(width=0),
    hovertemplate="<br>".join([
        "%{y} điểm",
    ]),
    name="Mức tăng nhỏ nhất"
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

fig.add_hline(
    y=0,
    line_width=0.1,
    # line_dash="dash",
    line_color="gray"
)

fig.update_layout(
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="white",
        font_size=20,
    ),
    autosize=False,
    height=450,

    font=dict(
        size=16
    ),

    margin=dict(
        t=0
    ),

    plot_bgcolor=BACKGROUND_COLOR,
    paper_bgcolor=BACKGROUND_COLOR,
)
fig = set_legend_at_top(fig)

fig.update_yaxes(
    range=[min(vn_index_last_5_days.min_change.min(), vn_index_last_day.change.min()),
           max(vn_index_last_5_days.max_change.max(), vn_index_last_day.change.max())]
)

fig.update_layout(
    yaxis_visible=False,
    yaxis_showticklabels=False,
    showlegend=False,
)

fig = set_axis_title_size(fig)
fig = disable_zoom(fig)
fig = set_axis_tick_color(fig)
st.plotly_chart(fig, config=CONFIG, use_container_width=True)

# Thanh khoản thị trường
st.header('Thanh khoản thị trường')
sector_amount = st.slider('Số lượng nhóm ngành', 0, 10, 5)
top_n_sector = latest_liquidity.query(
    'resolution == "market"'
    'or resolution == "sector"'
).sort_values(
    by='liquidity',
    ascending=False
)[['name']].values.flatten().tolist()[:sector_amount + 1]

if not st.checkbox('So sánh với thị trường', value=False):
    top_n_sector.remove('Thị trường')

market_and_top_n_liquidity = all_liquidity_same_period[all_liquidity_same_period['name'].isin(top_n_sector)]
market_and_top_n_liquidity['liquidity'] = market_and_top_n_liquidity['liquidity'] * 1000000000

fig_liquidity = px.line(
    data_frame=market_and_top_n_liquidity,
    x="indexed_timestamp_",
    y="liquidity",
    color="name",
    markers=True,
    line_shape="spline",
    hover_data={
        'indexed_timestamp_': True,
        'name': True,
        'liquidity': ':.2f'
    },
    labels={"indexed_timestamp_": "Ngày", "liquidity": "Thanh khoản", "name": "Ngành"},
    height=450
)

fig_liquidity.update_traces(
    hovertemplate='<br>Thanh khoản: %{y}',
)

fig_liquidity = line_chart_config(
    fig=fig_liquidity,
    marker_size=12,
    axis_title_size=12,
    tool_tip_font_size=14
)

fig_liquidity.update_traces(
    selector={"name": "Thị trường"},
    line={"dash": "dash"},
    mode="markers+lines",
)

fig_liquidity.update_layout(
    yaxis_title=None
)

# fig_liquidity.update_traces(
#     mode="markers+lines",
#     hovertemplate="<br>".join([
#         "Thời gian: %{x}",
#         "Ngành : %{name}",
#         "Thanh khoản: %{y}",
#     ]),
# )

fig_liquidity = set_legend_at_top(fig_liquidity)

st.plotly_chart(fig_liquidity, config=CONFIG, use_container_width=True)


# Net buy net sell
st.header('Tỉ lệ mã tăng giảm giá')

price_change_ratio = load_price_change_ratio()
# top_percentage_value = st.slider('Top % giao dịch', 0, 100, 60, step=5, format="%d%%")
min_symbol_count = st.slider('Số lượng mã tối thiểu trong ngành', 0, 20, 5, step=1, key='min_symbol_price_change_ratio')
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
    "price_increased_sum": "Tăng giá",
    "price_unchanged_mean": "Giá giữ nguyên",
    "no_price_info_mean": "Không có dữ liệu",
    "price_decreased_sum": "Giảm giá"
}


fig_price_change_ratio = go.Figure(
    go.Bar(
        x=price_change_ratio["price_increased_mean"],
        y=price_change_ratio["name"],
        marker_color=CYAN,
        orientation="h",
        name="Tăng"
    )
)

fig_price_change_ratio.add_trace(
    go.Bar(
        x=price_change_ratio["price_unchanged_mean"],
        y=price_change_ratio["name"],
        marker_color=AMBER,
        orientation="h",
        name="Giữ nguyên"
    )
)

fig_price_change_ratio.add_trace(
    go.Bar(
        x=price_change_ratio["no_price_info_mean"],
        y=price_change_ratio["name"],
        marker_color=LIGHT_GRAY,
        orientation="h",
        name="Không có thông tin"
    )
)

fig_price_change_ratio.add_trace(
    go.Bar(
        x=price_change_ratio["price_decreased_mean"],
        y=price_change_ratio["name"],
        marker_color=LIGHT_RED,
        orientation="h",
        name="Giảm"
    )
)

fig_price_change_ratio.update_yaxes(
    tickvals=price_change_ratio["name"],
    ticktext="   " + price_change_ratio["name"],
    ticklabelposition="inside",
    tickfont=dict(color=LIGHT_BLACK, size=10),
)

fig_price_change_ratio.update_xaxes(
    tickformat=',.0%',
    range=[0, 1],
    visible=False,
    tickfont=dict(size=10),
    side="top"
)

fig_price_change_ratio.update_layout(
    barmode="stack",
    showlegend=True,
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
    height=800
)

fig_price_change_ratio = axis_percentage(fig_price_change_ratio, axis='x')
fig_price_change_ratio = set_legend_at_top(fig_price_change_ratio)
fig_price_change_ratio = change_background_color(fig_price_change_ratio)
fig_price_change_ratio = disable_zoom(fig_price_change_ratio)

st.plotly_chart(fig_price_change_ratio, config=CONFIG, use_container_width=True)

# Net buy net sell
st.header('Tỉ lệ mã mua bán ròng')

net_buy_sell = load_net_buy_sell()
top_percentage_value = st.slider('Top % giao dịch', 0, 100, 60, step=5, format="%d%%")
min_symbol_count = st.slider('Số lượng mã tối thiểu trong ngành', 0, 20, 3, step=1, key='min_symbol_net_buy_sell')
net_buy_sell = net_buy_sell.query(
    f'symbol_count > {min_symbol_count}'
)
net_buy_sell = net_buy_sell.sort_values(
    by='symbol_count',
    ascending=False
).head(int(len(net_buy_sell) * top_percentage_value / 100))

net_buy_sell = net_buy_sell.sort_values(by='net_buy_ratio')

fig_top_net_sell = go.Figure(
    go.Bar(
        x=net_buy_sell["net_buy_ratio"],
        y=net_buy_sell["nganhcap3"],
        marker_color=CYAN,
        orientation="h",
        name="Mua ròng"
    )
)

fig_top_net_sell.add_trace(
    go.Bar(
        x=net_buy_sell["neutral_ratio"],
        y=net_buy_sell["nganhcap3"],
        marker_color=LIGHT_GRAY,
        orientation="h",
        name="Trung tính"
    )
)

fig_top_net_sell.add_trace(
    go.Bar(
        x=net_buy_sell["net_sell_ratio"],
        y=net_buy_sell["nganhcap3"],
        marker_color=LIGHT_RED,
        orientation="h",
        name="Bán ròng"
    )
)

fig_top_net_sell.update_yaxes(
    tickvals=net_buy_sell["nganhcap3"],
    ticktext="   " + net_buy_sell["nganhcap3"],
    ticklabelposition="inside",
    tickfont=dict(color=LIGHT_BLACK, size=10),
)

fig_top_net_sell.update_xaxes(
    tickformat=',.0%',
    range=[0, 1],
    visible=False,
    tickfont=dict(size=10),
    side="top"
)

fig_top_net_sell.update_layout(
    barmode="stack",
    showlegend=True,
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
    height=800
)

fig_top_net_sell = axis_percentage(fig_top_net_sell, axis='x')
fig_top_net_sell = set_legend_at_top(fig_top_net_sell)
fig_top_net_sell = change_background_color(fig_top_net_sell)
fig_top_net_sell = disable_zoom(fig_top_net_sell)

st.plotly_chart(fig_top_net_sell, config=CONFIG, use_container_width=True)

st.header('Thanh khoản so với quá khứ')
past_value = st.radio(
    "Thời gian",
    ["Hôm qua", "Trung bình 7 ngày"],
    index=0,
    horizontal=True
)

compare_col = 'liquidity_vs_1d_ago' if past_value == 'Hôm qua' else 'liquidity_vs_avg_7d_ago'
top_sector_vs_avg_7d_ago = st.slider('Top % giao dịch', 0, 100, 60, step=5, format="%d%%",
                                     key='top_sector_vs_avg_7d_ago')
lastest_liquidity = load_lastest_liquidity()

lastest_liquidity = lastest_liquidity.fillna(0).query('resolution == "sector"').sort_values(
    by='liquidity',
    ascending=False
).head(int(len(lastest_liquidity) * top_sector_vs_avg_7d_ago / 100))

lastest_liquidity = lastest_liquidity.sort_values(by=compare_col)
lastest_liquidity["color"] = np.where(lastest_liquidity[compare_col] < 0, LIGHT_RED, CYAN)

max_abs_liquidity = max(lastest_liquidity[compare_col].min(), lastest_liquidity[compare_col].max(), key=abs) * 1.2

fig_liquidity_vs_1d_ago = go.Figure(
    go.Bar(
        x=[max_abs_liquidity] * len(lastest_liquidity),
        y=lastest_liquidity["name"],
        text=[gen_percentage_string(i) for i in lastest_liquidity[compare_col]],
        textposition="inside",
        textfont=dict(color="black"),
        orientation="h",
        marker_color=INVISIBLE,
    )
)

fig_liquidity_vs_1d_ago.add_trace(
    go.Bar(
        x=abs(lastest_liquidity[compare_col]),
        y=lastest_liquidity["name"],
        marker_color=lastest_liquidity["color"],
        orientation="h",
    )
)

fig_liquidity_vs_1d_ago.update_yaxes(
    tickvals=lastest_liquidity["name"],
    ticktext="   " + lastest_liquidity["name"],
    ticklabelposition="inside",
    tickfont=dict(color=LIGHT_BLACK, size=10),
)

fig_liquidity_vs_1d_ago.update_xaxes(
    tickformat=',.0%',
    range=[0, max_abs_liquidity],
    visible=False,
    tickfont=dict(size=10),
    side="top"
)

fig_liquidity_vs_1d_ago.update_layout(
    barmode="overlay",
    showlegend=False,
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
    height=1000
)

fig_liquidity_vs_1d_ago = change_background_color(fig_liquidity_vs_1d_ago)
fig_liquidity_vs_1d_ago = disable_zoom(fig_liquidity_vs_1d_ago)
st.plotly_chart(fig_liquidity_vs_1d_ago, config=CONFIG, use_container_width=True)


# Top ngành có số mã tăng giá giảm giá
st.header('Top ngành có số mã tăng giá')
top_price_increase_amount = st.slider('Số lượng ngành tăng', 0, 20, 10, step=5)
top_price_increase = latest_price_metric.query('name != "market"').sort_values(
    by='price_increased_sum',
    ascending=False
).head(top_price_increase_amount)

top_price_increase = top_price_increase.sort_values(by='price_increased_sum')

fig_price_increase = go.Figure(
    go.Bar(
        x=[top_price_increase["price_increased_sum"].max() * 1.2] * top_price_increase_amount,
        y=top_price_increase["name"],
        text=top_price_increase["price_increased_sum"],
        textposition="inside",
        textfont=dict(color="black"),
        orientation="h",
        marker_color=INVISIBLE,
    )
)

fig_price_increase.add_trace(
    go.Bar(
        x=top_price_increase["price_increased_sum"],
        y=top_price_increase["name"],
        marker={'color': top_price_increase["price_increased_sum"], 'colorscale': [LIGHT_CYAN, CYAN]},
        orientation="h",
    )
)

fig_price_increase.update_yaxes(
    tickvals=top_price_increase["name"],
    ticktext="   " + top_price_increase["name"],
    ticklabelposition="inside",
    tickfont=dict(color=LIGHT_BLACK, size=12),
)

fig_price_increase.update_xaxes(
    # tickformat=',.0%',
    range=[0, top_price_increase["price_increased_sum"].max() * 1.2],
    visible=False,
    tickfont=dict(size=12),
    side="top"
)

fig_price_increase.update_layout(
    barmode="overlay",
    showlegend=False,
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
    height=450
)

fig_price_increase = change_background_color(fig_price_increase)
fig_price_increase = disable_zoom(fig_price_increase)
st.plotly_chart(fig_price_increase, config=CONFIG, use_container_width=True)

st.header('Top ngành có số mã giảm giá')
top_price_decrease_amount = st.slider('Số lượng ngành giảm', 0, 20, 10, step=5)
top_price_decrease = latest_price_metric.query('name != "market"').sort_values(
    by='price_decreased_sum',
    ascending=False
).head(top_price_decrease_amount)

top_price_decrease = top_price_decrease.sort_values(by='price_decreased_sum')

fig_price_decrease = go.Figure(
    go.Bar(
        x=[top_price_increase["price_decreased_sum"].max() * 1.2] * top_price_increase_amount,
        y=top_price_decrease["name"],
        text=top_price_decrease["price_decreased_sum"],
        textposition="inside",
        textfont=dict(color="black"),
        orientation="h",
        marker_color=INVISIBLE,
    )
)

fig_price_decrease.add_trace(
    go.Bar(
        x=top_price_decrease["price_decreased_sum"],
        y=top_price_decrease["name"],
        marker={'color': top_price_decrease["price_decreased_sum"], 'colorscale': [SUPER_LIGHT_RED, LIGHT_RED]},
        orientation="h",
    )
)

fig_price_decrease.update_yaxes(
    tickvals=top_price_decrease["name"],
    ticktext="   " + top_price_decrease["name"],
    ticklabelposition="inside",
    tickfont=dict(color=LIGHT_BLACK, size=12),
)

fig_price_decrease.update_xaxes(
    # tickformat=',.0%',
    range=[0, top_price_increase["price_decreased_sum"].max() * 1.2],
    visible=False,
    tickfont=dict(size=12),
    side="top"
)

fig_price_decrease.update_layout(
    barmode="overlay",
    showlegend=False,
    margin=dict(
        t=10,
        b=20,
        pad=5
    ),
    height=450,
)

fig_price_decrease = change_background_color(fig_price_decrease)
fig_price_decrease = disable_zoom(fig_price_decrease)
st.plotly_chart(fig_price_decrease, config=CONFIG, use_container_width=True)
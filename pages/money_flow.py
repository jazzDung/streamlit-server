import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
import logging

logger = logging.getLogger('tcpserver')


st.set_page_config(page_title="Mapping Demo", page_icon="🌍", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

conn = st.experimental_connection("postgresql", type="sql")


# for row in df.itertuples():
#     st.write(f"{row.name} has a :{row.pet}:")


# chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])


dict_data = st.experimental_get_query_params()
try:
        
    from_date = str(dict_data['from_date'][0])
    to_date = str(dict_data['to_date'][0])
    # symbol = str(dict_data['symbol'][0])

except Exception as e:
    from_date = 'None'
    to_date = 'None'
    # symbol = 'None'


if from_date == 'None' or from_date == '':
    from_date = '10-09-2023'
    to_date = '10-11-2023'

df = conn.query(f"select liquidity from liquidity_all_resolution where resolution ='market' and indexed_timestamp_::date >= '{from_date}' and indexed_timestamp_::date <= '{to_date}' order by indexed_timestamp_ ASC", ttl="10m")

# date_list = df['indexed_timestamp'].unique().tolist()


st.line_chart(df)
import streamlit as st
import time
import numpy as np
import pandas as pd 

st.set_page_config(page_title="Plotting Demo", page_icon="📈", initial_sidebar_state="collapsed")

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
df = conn.query("select liquidity from liquidity_all_resolution where resolution = 'market' limit 100 ", ttl="10m")


# for row in df.itertuples():
#     st.write(f"{row.name} has a :{row.pet}:")


# chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.line_chart(df)

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
# st.button("Re-run")
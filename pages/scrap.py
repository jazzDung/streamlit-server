# hour_to_filter = st.slider('hour', 0, 23, 17)
# # filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
#
# st.text_input("Your name", key="name")
#
# # You can access the value at any point with:
# st.session_state.name
#
# option = st.selectbox(
#     'Which number do you like best?',
#     all_liquidity_same_period['liquidity'])
#
# 'You selected: ', option
#
# # Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     'Nhóm ngành',
#     ('Email', 'Home phone', 'Mobile phone')
# )
#
# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")
#
#
# # Add a placeholder
# latest_iteration = st.sidebar.empty()
# bar = st.sidebar.progress(0)
#
# for i in range(100):
#     # Update the progress bar with each iteration.
#     bar.progress(i + 1, text=f'Làm mới dữ liệu {str(i + 1) + "%" if i + 1 < 100 else "hoàn tất"}')
#     time.sleep(0.05)
#
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.dataframe(all_liquidity_same_period.style.highlight_max(axis=0))
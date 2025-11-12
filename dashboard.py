import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from PIL import Image
import os


def load_data():
    """Load data from SQLite database"""
    conn = sqlite3.connect('small_tables/small_tables.db')
    tables = {}

    # Get list of all tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()

    # Load each table into a dictionary
    for (table_name,) in table_names:
        tables[table_name] = pd.read_sql(f"SELECT * FROM {table_name}", conn)

    conn.close()
    return tables


def main():
    st.set_page_config(page_title="Hotel Analysis Dashboard", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page",
                            ["Overview", "Hotel Analysis", "Data Tables", "Documentation"])

    # Load data
    tables = load_data()

    if page == "Overview":
        st.title("Hotel Analysis Dashboard")

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'query_1_sample' in tables:
                total_hotels = tables['query_1_sample']['num_hotels'].sum()
                st.metric("Total Hotels", f"{total_hotels:,}")

        with col2:
            if 'query_2_sample' in tables:
                luxury_hotels = tables['query_2_sample']['num_high_rated_hotels'].sum()
                st.metric("Luxury Hotels", f"{luxury_hotels:,}")

        with col3:
            if 'query_6_sample' in tables:
                total_cities = tables['query_6_sample']['total_cities'].sum()
                st.metric("Total Cities", f"{total_cities:,}")

        # Display main visualizations
        if os.path.exists('static/hotels_by_county_pie_chart.png'):
            st.image('static/hotels_by_county_pie_chart.png')

    elif page == "Hotel Analysis":
        st.title("Detailed Hotel Analysis")

        # Show interactive plots
        if 'query_1_sample' in tables:
            df = tables['query_1_sample']
            fig = px.bar(df,
                         x='countyName',
                         y='num_hotels',
                         title='Hotels by Country')
            st.plotly_chart(fig)

            fig = px.scatter(df,
                             x='num_hotels',
                             y='avg_rating',
                             hover_data=['countyName'],
                             title='Hotels vs Average Rating')
            st.plotly_chart(fig)

    elif page == "Data Tables":
        st.title("Raw Data Tables")

        # Table selector
        table_name = st.selectbox("Select a table", list(tables.keys()))

        # Display selected table
        if table_name in tables:
            st.write(f"### {table_name}")
            st.dataframe(tables[table_name])

            # Download button
            csv = tables[table_name].to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f'{table_name}.csv',
                mime='text/csv',
            )

    elif page == "Documentation":
        st.title("Documentation")

        if os.path.exists('documentation/database_info.md'):
            with open('documentation/database_info.md', 'r') as f:
                st.markdown(f.read())


if __name__ == "__main__":
    main()

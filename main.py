import duckdb
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from datetime import datetime

# Create necessary directories
dirs = ['query_databases', 'small_tables', 'documentation', 'static']
for dir_name in dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# Define SQL queries
query1 = """
WITH filtered_hotels AS (
    SELECT 
        countyName, 
        HotelCode, 
        CASE 
            WHEN HotelRating = 'FiveStar' THEN 5
            WHEN HotelRating = 'FourStar' THEN 4
            WHEN HotelRating = 'ThreeStar' THEN 3
            WHEN HotelRating = 'TwoStar' THEN 2
            WHEN HotelRating = 'OneStar' THEN 1
        END AS rating
    FROM read_csv_auto('hotels.csv', ignore_errors=true) 
    WHERE HotelRating NOT IN ('All')
),
county_hotels AS (
    SELECT 
        countyName, 
        COUNT(DISTINCT HotelCode) AS num_hotels,
        ROUND(AVG(rating), 2) AS avg_rating
    FROM filtered_hotels
    GROUP BY countyName
),
total_hotels AS (
    SELECT SUM(num_hotels) AS world_total_hotels
    FROM county_hotels
)
SELECT 
    countyName, 
    num_hotels,
    avg_rating,
    ROUND(num_hotels * 100.0 / world_total_hotels, 2) AS percent_of_world_hotels
FROM county_hotels, total_hotels
ORDER BY percent_of_world_hotels DESC
"""

query2 = """
WITH high_rated_hotels AS (
    SELECT 
        countyName,
        COUNT(DISTINCT HotelCode) as luxury_hotels
    FROM read_csv_auto('hotels.csv', ignore_errors=true)
    WHERE HotelRating IN ('FiveStar', 'All')
    GROUP BY countyName
),
total_luxury AS (
    SELECT SUM(luxury_hotels) as world_total_luxury
    FROM high_rated_hotels
)
SELECT 
    h.countyName,
    h.luxury_hotels as num_high_rated_hotels,
    ROUND(h.luxury_hotels * 100.0 / t.world_total_luxury, 2) as percent_of_world_luxury,
    t.world_total_luxury as total_luxury_hotels_worldwide
FROM high_rated_hotels h, total_luxury t
ORDER BY h.luxury_hotels DESC;
"""

query3 = """
WITH website_stats AS (
    SELECT 
        COUNT(DISTINCT HotelCode) as total_hotels,
        COUNT(DISTINCT CASE WHEN HotelWebsiteUrl IS NOT NULL THEN HotelCode END) as hotels_with_urls,
        COUNT(DISTINCT CASE WHEN LOWER(HotelWebsiteUrl) LIKE '%booking%' THEN HotelCode END) as hotels_with_booking
    FROM read_csv_auto('hotels.csv', ignore_errors=true)
)
SELECT 
    hotels_with_booking as num_hotels_with_booking,
    total_hotels as total_hotels,
    hotels_with_urls as hotels_with_urls,
    ROUND(hotels_with_booking * 100.0 / total_hotels, 2) as percent_of_all_hotels,
    ROUND(hotels_with_booking * 100.0 / hotels_with_urls, 2) as percent_of_hotels_with_urls
FROM website_stats;
"""

query4 = """
WITH country_hotel_counts AS (
    SELECT 
        countyName, 
        COUNT(DISTINCT HotelCode) AS num_hotels
    FROM read_csv_auto('hotels.csv', ignore_errors=true)
    GROUP BY countyName
)
SELECT 
    countyName, 
    num_hotels,
    RANK() OVER (ORDER BY num_hotels DESC) AS rank_in_world,
    ROUND(100.0 * num_hotels / FIRST_VALUE(num_hotels) OVER (ORDER BY num_hotels DESC), 2) AS percent_of_top_country
FROM country_hotel_counts
ORDER BY rank_in_world;
"""

query5 = """
WITH high_rated_hotels AS (
    SELECT 
        countyName, 
        HotelName,
        HotelRating,
        LENGTH(Attractions) - LENGTH(REPLACE(Attractions, ',', '')) + 1 AS num_attractions
    FROM read_csv_auto('hotels.csv', ignore_errors=true)
    WHERE HotelRating = 'FiveStar' AND TRIM(Attractions) != ''
),
country_attractions AS (
    SELECT 
        countyName,
        COUNT(DISTINCT HotelName) AS num_high_rated_hotels,
        SUM(num_attractions) AS total_attractions,
        ROUND(AVG(num_attractions), 2) AS avg_attractions_per_hotel
    FROM high_rated_hotels
    GROUP BY countyName
)
SELECT 
    countyName,
    num_high_rated_hotels,
    total_attractions,
    avg_attractions_per_hotel
FROM country_attractions
ORDER BY total_attractions DESC
LIMIT 10;
"""

query6 = """
WITH country_city_hotel_avg AS (
    SELECT 
        countyName,
        COUNT(DISTINCT HotelCode) AS total_hotels,
        COUNT(DISTINCT cityName) AS total_cities,
        ROUND(1.0 * COUNT(DISTINCT HotelCode) / COUNT(DISTINCT cityName), 2) AS avg_hotels_per_city
    FROM read_csv_auto('hotels.csv', ignore_errors=true)
    WHERE TRIM(cityName) != ''  -- Exclude hotels without a valid city name
    GROUP BY countyName
)
SELECT 
    countyName,
    total_hotels,
    total_cities,
    avg_hotels_per_city
FROM country_city_hotel_avg
WHERE total_cities > 0  -- Exclude countries without cities
ORDER BY avg_hotels_per_city DESC
LIMIT 10;
"""


def create_pie_chart(df, value_column, title, filename, top_n=15):
    """
    Creates a pie chart visualization

    Args:
        df: DataFrame containing the data
        value_column: Column name for values to plot
        title: Title of the chart
        filename: Output file path
        top_n: Number of top items to show (rest grouped as 'Others')
    """
    # Take top N, group others
    top = df.head(top_n)
    others = pd.DataFrame({
        'countyName': ['Others'],
        value_column: [df.iloc[top_n:][value_column].sum()]
    })
    plot_df = pd.concat([top, others])

    plt.figure(figsize=(12, 8))
    plt.pie(plot_df[value_column],
            labels=plot_df['countyName'],
            autopct='%1.1f%%')
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_bar_chart(df, x_column, y_columns, title, filename, label_y1, label_y2):
    """
    Creates a bar chart visualization with two sets of bars

    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis
        y_columns: List of two column names for y-axis values
        title: Title of the chart
        filename: Output file path
        label_y1: Label for first y-axis
        label_y2: Label for second y-axis
    """
    x = np.arange(len(df[x_column]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width / 2, df[y_columns[0]], width, label=label_y1)
    bars2 = ax.bar(x + width / 2, df[y_columns[1]], width, label=label_y2)

    ax.set_xlabel('Countries')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_column], rotation=45, ha='right')
    ax.legend()

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class SQLOperationsHandler:
    def __init__(self):
        # Initialize DuckDB connection
        self.duck_conn = duckdb.connect('query_databases/main.db')
        self.cursor = self.duck_conn.cursor()

    def execute_and_save_query(self, query_number, query, description, columns, sample_method="random"):
        
        """
                Executes a query and saves both full and sampled results
                """
        print(f"\n{'=' * 50}")
        print(f"Executing Query {query_number}: {description}")
        print(f"SQL Query:\n{query}")

        try:
            # Execute query and get full results
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            # Convert to DataFrame
            full_df = pd.DataFrame(results, columns=columns)

            # Create sampled version (approximately 500 rows)
            if sample_method == "random":
                sampled_df = full_df.sample(n=min(500, len(full_df)), random_state=42)
            elif sample_method == "stratified":
                sampled_df = full_df.groupby(columns[0], group_keys=False) \
                    .apply(lambda x: x.sample(min(int(500 / len(full_df[columns[0]].unique())), len(x)))) \
                    .reset_index(drop=True)

            # Save full results in DuckDB
            table_name = f"query_{query_number}_results"
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Create table directly from the query
            self.cursor.execute(f"CREATE TABLE {table_name} AS {query}")

            # Save sampled results in DuckDB
            sample_table_name = f"query_{query_number}_sample"
            self.cursor.execute(f"DROP TABLE IF EXISTS {sample_table_name}")

            # Convert sampled DataFrame to a values list for insertion
            values = sampled_df.values.tolist()
            columns_str = ', '.join(columns)
            placeholders = ', '.join(['?' for _ in columns])

            # Create empty table with correct schema
            create_sample_table_query = f"""
                        CREATE TABLE {sample_table_name} AS 
                        SELECT * FROM {table_name} WHERE 1=0
                    """
            self.cursor.execute(create_sample_table_query)

            # Insert sampled data
            insert_query = f"INSERT INTO {sample_table_name} ({columns_str}) VALUES ({placeholders})"
            self.cursor.executemany(insert_query, values)

            print(f"Full results shape: {full_df.shape}")
            print(f"Sampled results shape: {sampled_df.shape}")
            return full_df, sampled_df

        except Exception as e:
            print(f"Error executing query {query_number}: {str(e)}")
            raise


    def transfer_to_sqlite(self, sqlite_db_path='small_tables/small_tables.db'):
        """
        Transfers all sampled tables from DuckDB to SQLite
        """
        # Get list of sample tables
        self.cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE '%sample%'
        """)
        sample_tables = self.cursor.fetchall()

        # Create SQLite connection
        sqlite_conn = sqlite3.connect(sqlite_db_path)

        # Transfer each sample table
        for (table_name,) in sample_tables:
            # Read data from DuckDB
            self.cursor.execute(f"SELECT * FROM {table_name}")
            data = self.cursor.fetchall()

            # Get column names
            self.cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """)
            columns = [col[0] for col in self.cursor.fetchall()]

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=columns)

            # Save to SQLite
            df.to_sql(table_name, sqlite_conn, if_exists='replace', index=False)

            print(f"Transferred table {table_name} to SQLite")

        sqlite_conn.close()

    def get_table_info(self, sqlite_db_path='small_tables/small_tables.db'):
            """
            Gets information about all tables in the SQLite database
            """
            sqlite_conn = sqlite3.connect(sqlite_db_path)
            cursor = sqlite_conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            table_info = []
            for (table_name,) in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_count = len(columns)
                column_names = [col[1] for col in columns]

                table_info.append({
                    'table_name': table_name,
                    'rows': row_count,
                    'columns': column_count,
                    'column_names': column_names
                })

            sqlite_conn.close()
            return table_info

    def generate_documentation(self):
            """
            Generates documentation about the database and visualizations
            """
            table_info = self.get_table_info()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('documentation/database_info.md', 'w') as f:
                f.write(f"# Database Information\nGenerated on: {timestamp}\n\n")
                f.write("## Tables Overview\n\n")
                for table in table_info:
                    f.write(f"### {table['table_name']}\n")
                    f.write(f"- Rows: {table['rows']}\n")
                    f.write(f"- Columns: {table['columns']}\n")
                    f.write("- Column Names: " + ", ".join(table['column_names']) + "\n\n")

                # Add visualization documentation
                f.write("\n## Visualizations\n\n")
                visualizations = {
                    'hotels_by_county_pie_chart.png': 'Shows distribution of hotels across countries',
                    'luxury_hotels_by_county_pie_chart.png': 'Shows distribution of luxury hotels',
                    'top_countries_by_hotels_pie_chart.png': 'Shows top countries by hotel count',
                    'top_countries_hotels_attractions.png': 'Compares hotels and attractions by country',
                    'top_cities_by_hotel_density.png': 'Shows hotel density in top cities'
                }

                for viz, description in visualizations.items():
                    f.write(f"### {viz}\n")
                    f.write(f"- {description}\n")
                    f.write("- Location: static/\n\n")


    def close(self):
        """Close the database connection"""
        self.duck_conn.close()

def create_visualizations(handler):
    """
    Creates all visualizations using data from the SQL queries
    """
    # Define queries with descriptions
    queries = [
        {
            "number": 1,
            "query": query1,
            "description": "Analysis of hotel distribution and ratings across countries",
            "columns": ['countyName', 'num_hotels', 'avg_rating', 'percent_of_world_hotels']
        },
        {
            "number": 2,
            "query": query2,
            "description": "Distribution of luxury hotels worldwide",
            "columns": ['countyName', 'num_high_rated_hotels', 'percent_of_world_luxury', 'total_luxury_hotels_worldwide']
        },
        {
            "number": 3,
            "query": query3,
            "description": "Analysis of hotel website presence and booking capabilities",
            "columns": ['num_hotels_with_booking', 'total_hotels', 'hotels_with_urls',
                        'percent_of_all_hotels', 'percent_of_hotels_with_urls']
        },
        {
            "number": 4,
            "query": query4,
            "description": "Country rankings by hotel presence",
            "columns": ['countyName', 'num_hotels', 'rank_in_world', 'percent_of_top_country']
        },
        {
            "number": 5,
            "query": query5,
            "description": "Analysis of attractions near high-rated hotels",
            "columns": ['countyName', 'num_high_rated_hotels', 'total_attractions', 'avg_attractions_per_hotel']
        },
        {
            "number": 6,
            "query": query6,
            "description": "Analysis of hotel density in cities",
            "columns": ['countyName', 'total_hotels', 'total_cities', 'avg_hotels_per_city']
        }
    ]

    # Process each query and create visualizations
    for query_info in queries:
        # Execute query and save results
        full_df, sampled_df = handler.execute_and_save_query(
            query_info["number"],
            query_info["query"],
            query_info["description"],
            query_info["columns"]
        )

        # Create visualizations based on query number
        if query_info["number"] == 1:
            create_pie_chart(full_df, 'percent_of_world_hotels',
                            'Distribution of Hotels by Country',
                            'static/hotels_by_county_pie_chart.png')
        elif query_info["number"] == 2:
            create_pie_chart(full_df, 'percent_of_world_luxury',
                            'Distribution of Luxury Hotels by Country',
                            'static/luxury_hotels_by_county_pie_chart.png')
        elif query_info["number"] == 4:
            create_pie_chart(full_df, 'percent_of_top_country',
                            'Top Countries by Hotel Distribution',
                            'static/top_countries_by_hotels_pie_chart.png')
        elif query_info["number"] == 5:
            create_bar_chart(full_df, 'countyName',
                            ['num_high_rated_hotels', 'total_attractions'],
                            'Top Countries by High-Rated Hotels and Attractions',
                            'static/top_countries_hotels_attractions.png',
                            'High-Rated Hotels', 'Total Attractions')
        elif query_info["number"] == 6:
            create_bar_chart(full_df, 'countyName',
                            ['total_hotels', 'total_cities'],
                            'Hotel and City Distribution by Country',
                            'static/top_cities_by_hotel_density.png',
                            'Total Hotels', 'Total Cities')

    # Transfer sampled tables to SQLite
    handler.transfer_to_sqlite()



if __name__ == "__main__":
    # Initialize handler
    handler = SQLOperationsHandler()

    # Process queries and create visualizations
    create_visualizations(handler)

    # Generate documentation
    handler.generate_documentation()

    # Write table information
    table_info = handler.get_table_info()
    with open('documentation/table_info.txt', 'w') as f:
        for table in table_info:
            f.write(f"Table: {table['table_name']}\n")
            f.write(f"Rows: {table['rows']}\n")
            f.write(f"Columns: {table['columns']}\n")
            f.write("Column Names: " + ", ".join(table['column_names']) + "\n\n")

    # Close connections
    handler.close()

    print("Processing complete. Run 'streamlit run dashboard.py' to start the dashboard.")

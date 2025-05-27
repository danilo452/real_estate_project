import streamlit as st
import pandas as pd 
import pydeck as pdk
import plotly.express as px

# Page configuration
st.set_page_config(layout="wide", page_title="Real Estate Report", page_icon="🏠")
st.header("Real Estate Report per US Market 🏠", divider='blue')
st.sidebar.title("Main Report Filters")

# Constants
MAPA_COLUMNS_MAPPING = {
    'irr': ('IRR', '%'),
    'equity_multiple': ('Equity Multiple', 'x'),
    'current_price': ('Current Price', 'USD'),
    'loan': ('Loan', 'USD'),
    'equity': ('Equity', 'USD'),
    'market_cagr': ('Market CAGR', '%'),
    'noi_cap_rate_compounded': ('NOI Cap Rate Compounded', '%'),
    'operation_cashflow': ('Operation Cashflow', '%'),
    'market_cap_appreciation_bp': ('Market Cap Appreciation', 'bp'),
    'npv': ('Net Present Value', 'USD'),
    'npv/equity': ('NPV/Equity', 'x'),
    'demand_vs_supply': ('Demand vs Supply', '%'),
    'demand_yoy_growth': ('Demand YoY Growth', '%'),
    'supply_yoy_growth': ('Supply YoY Growth', '%')
}
MAP_STYLE = "mapbox://styles/mapbox/streets-v11"
VIEW_STATE = pdk.ViewState(
    longitude=-99, latitude=38.83, zoom=3.4, min_zoom=2, max_zoom=7, pitch=75, bearing=23
)

# Load data
def load_data():
    def process_market_data(file_path, date_column):
        data = pd.read_csv(file_path, index_col=0, parse_dates=True).drop(columns=[date_column])
        data['state'] = data['market'].str.split(',').str[1]
        data['city'] = data['market'].str.split(',').str[0]
        return data

    cashflow = process_market_data('data/cashflows.csv', 'date_cashflow')
    equilibrium = process_market_data('data/equilibriums.csv', 'date_calculated')
    table = process_market_data('data/tables.csv', 'date_calculated')

    summary = pd.read_csv('data/summaries.csv', index_col=0, parse_dates=True).drop(columns=['date_calculated'])
    general = pd.read_csv('data/general_parameters.csv', index_col=0)
    general_2 = pd.read_csv('data/general_parameters_2.csv', index_col=0)

    return cashflow, equilibrium, summary, table, general, general_2

cashflow, equilibrium, summary, table, general, general_2 = load_data()

# Sidebar Filters
states = summary['state'].unique()
classes = summary['slice'].unique()
horizons = summary['horizon'].unique()

def sidebar_filters():
    st.session_state.slice = st.sidebar.selectbox('Select Class', options=classes, index=0)
    st.session_state.horizon = st.sidebar.selectbox('Select Horizon (Last Years)', options=horizons, index=0)
    population = st.sidebar.selectbox(
        'Select Population',
        options=['All', '+100K', '+500K', '+1M', '+2M', '+3M', '+5M', '+7M', '+10M'],
        index=2
    ).replace('+', '').replace('.', '').replace('M', '000000').replace('K', '000').replace('All', '0')
    st.session_state.population = int(population)
    filtro_columnas_mapa = st.sidebar.selectbox('Aspect to classify', options=list(MAPA_COLUMNS_MAPPING.keys()), index=0)
    st.session_state.filtro_columnas_mapa = filtro_columnas_mapa
    if st.sidebar.button('Reset Filters'):
        st.rerun()

sidebar_filters()

# Filter Summary Data
def filter_summary():
    filtered = summary[
        (summary['slice'] == st.session_state.slice) &
        (summary['state'].isin(states)) &
        (summary['population'] >= st.session_state.population) &
        (summary['horizon'] == st.session_state.horizon)
    ].copy()
    if filtered.empty:
        st.error('No data available for the selected filters')
        st.stop()
    return filtered

summary_filtered = filter_summary()

# Map Layer Configurations
def create_map_layers(data):
    irr_layer = pdk.Layer(
        "ColumnLayer",
        data=data,
        get_position=["longitude", "latitude"],
        get_elevation="bar_height",
        elevation_scale=2500,
        radius=20000,
        get_fill_color="color",
        pickable=True,
        extruded=True,
        auto_highlight=True
    )

    population_layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["longitude", "latitude"],
        get_radius="log_population",
        radius_scale=90,
        get_fill_color=[55, 8, 94, 60],
        pickable=True
    )

    return [irr_layer, population_layer]

layers = create_map_layers(summary_filtered)

# Render Map
st.pydeck_chart(
    pdk.Deck(
        map_style=MAP_STYLE,
        initial_view_state=VIEW_STATE,
        layers=layers,
        tooltip={
            "html": """<b>State:</b> {state}<br/><b>City:</b> {city}<br/><b>Value:</b> {value_show}<br/><b>Population:</b> {population_millions}""",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
                "fontSize": "14px",
                "padding": "10px",
                "borderRadius": "15px"
            }
        }
    ),
    use_container_width=True
)

# Display Market Summary
st.subheader('US Markets Summary', divider='blue')
data_show = summary_filtered[['market', 'current_price', 'market_cagr', 'noi_cap_rate_compounded',
                               'fixed_interest_rate', 'operation_cashflow', 'market_cap_appreciation_bp', 'irr', 
                               'npv', 'npv/equity', 'equity_multiple', 'demand_vs_supply',
                               'demand_yoy_growth', 'supply_yoy_growth']]
data_show = data_show.sort_values(st.session_state.filtro_columnas_mapa, ascending=False)
st.dataframe(data_show.set_index('market'), use_container_width=True)

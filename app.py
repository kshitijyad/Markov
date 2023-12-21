import streamlit as st
from hmmlearn import hmm
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

def calculate_log_returns_and_training_data(data, metrics, training_end_date):
    """
    Calculate log returns for the specified metrics in a stock data DataFrame and 
    segment the data into a training set.

    Args:
    data (DataFrame): The stock data.
    metrics (list): List of column names for which to calculate log returns.
    training_end_date (str): The end date for the training data.

    Returns:
    DataFrame: A DataFrame with log returns for the entire dataset.
    ndarray: A NumPy array with reshaped training data.
    """
    log_returns = pd.DataFrame()

    for metric in metrics:
        if metric in data.columns:
            log_returns[metric] = np.log(1 + data[[metric]].pct_change())
        else:
            print(f"Metric '{metric}' not found in data.")

    log_returns.dropna(inplace=True)

    # Define the training data
    training_data = log_returns[log_returns.index < pd.to_datetime(training_end_date)]
    training_data_reshaped = training_data.values

    return log_returns, training_data_reshaped

def calculate_hmm_trading_returns(initial_cash,data, buy_state):
    cash = initial_cash
    position = 0.0
    cash_positions = [initial_cash]  # Start with initial cash value
    for i in range(len(data)):
        current_state = data.iloc[i]['Predicted_State']
        current_price = data.iloc[i]['Adj Close']

        # Trading logic
        if current_state in buy_state and cash > 0:
            position = cash / current_price
            cash = 0.0
        elif current_state not in buy_state and position > 0:
            cash = position * current_price
            position = 0.0

        # Update cash positions for each day
        # If in position, show unrealized profit/loss; otherwise, show last realized cash
        current_position_value = position * current_price if position > 0 else cash
        cash_positions.append(current_position_value)

    # Calculate final value and returns
    final_value = cash + position * data.iloc[-1]['Adj Close']
    total_return = (final_value / initial_cash - 1) * 100
    return total_return, cash_positions

def calculate_cagr(start_value, end_value, start_date, end_date):
    number_of_years = (end_date - start_date).days / 365.25
    return (end_value / start_value) ** (1 / number_of_years) - 1

def calculate_max_drawdown(portfolio_values):
    # Convert to numpy array for easier manipulation
    portfolio_values = np.array(portfolio_values)

    # Calculate cumulative returns
    cumulative_returns = portfolio_values / portfolio_values[0]

    # Track the running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Calculate drawdown
    drawdown = (running_max - cumulative_returns) / running_max

    # Maximum drawdown
    max_drawdown = np.max(drawdown)
    return max_drawdown
def calculate_hmm_trading_returns(initial_cash,data, buy_state):
        cash = initial_cash
        position = 0.0
        cash_positions = [initial_cash]  # Start with initial cash value
        for i in range(len(data)):
            current_state = data.iloc[i]['Predicted_State']
            current_price = data.iloc[i]['Adj Close']

            # Trading logic
            if current_state in buy_state and cash > 0:
                position = cash / current_price
                cash = 0.0
            elif current_state not in buy_state and position > 0:
                cash = position * current_price
                position = 0.0

            # Update cash positions for each day
            # If in position, show unrealized profit/loss; otherwise, show last realized cash
            current_position_value = position * current_price if position > 0 else cash
            cash_positions.append(current_position_value)

        # Calculate final value and returns
        final_value = cash + position * data.iloc[-1]['Adj Close']
        total_return = (final_value / initial_cash - 1) * 100
        return total_return, cash_positions

def calculate_cagr(start_value, end_value, start_date, end_date):
    number_of_years = (end_date - start_date).days / 365.25
    return (end_value / start_value) ** (1 / number_of_years) - 1

def calculate_max_drawdown(portfolio_values):
    # Convert to numpy array for easier manipulation
    portfolio_values = np.array(portfolio_values)

    # Calculate cumulative returns
    cumulative_returns = portfolio_values / portfolio_values[0]

    # Track the running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Calculate drawdown
    drawdown = (running_max - cumulative_returns) / running_max

    # Maximum drawdown
    max_drawdown = np.max(drawdown)
    return max_drawdown
st.title("Stock Market Analysis Using HMM")


    

@st.cache_data
def fetch_and_process_data(stock, start_date, end_date, training_date, metrics, n_components):
    data = yf.download(stock, start=start_date, end=end_date, period="1mo")
    #st.experimental_rerun()
    #data=data.resample('D').ffill()
    combined_log_returns, training_data_reshaped = calculate_log_returns_and_training_data(data, metrics, training_date)
    combined_log_returns_reshaped = combined_log_returns.values
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100000)
    model.fit(np.array(training_data_reshaped).reshape(-1, 1))

    # Predict states for the entire dataset
    predicted_states = model.predict(combined_log_returns_reshaped)

    # Transition matrix and means/variances of each state
    print("Transition matrix")
    print(model.transmat_)
    print("Means and variances of each state")
    for i in range(n_components):
        print(f"State {i}: mean = {model.means_[i][0]}, variance = {np.diag(model.covars_[i])[0]}")

    # Add predicted states to the original data
    data_aligned = data[len(data) - len(predicted_states):].assign(Predicted_State=predicted_states)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_aligned.index, y=data_aligned['Adj Close'], line=dict(color='lightgrey'), name='Adjusted Close'))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for state in range(n_components):
        state_data = data_aligned[data_aligned['Predicted_State'] == state]
        fig.add_trace(go.Scatter(x=state_data.index, y=state_data['Adj Close'], mode='markers', marker=dict(color=colors[state], size=5), name=f'State {state}'))
    fig.update_layout(title='Stock Adjusted Close Prices with Market States', xaxis_title='Date', yaxis_title='Adjusted Close Price', hovermode='x unified')

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
    return data_aligned,data

def initialize_state():
    st.write("Initializing")
    if 'stock' not in st.session_state:
        st.session_state['stock'] = 'SPY'
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = pd.to_datetime("1995-07-10")
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = pd.to_datetime("now")
    if 'training_date' not in st.session_state:
        st.session_state['training_date'] = pd.to_datetime("2018-01-01")
    if 'metrics' not in st.session_state:
        st.session_state['metrics'] = ['Adj Close']
    if 'n_components' not in st.session_state:
        st.session_state['n_components'] = 2
    if 'buy_state' not in st.session_state:
        st.session_state['buy_state'] = [1]
    if 'initial_cash' not in st.session_state:
        st.session_state['initial_cash'] = 10000.0

initialize_state()

with st.sidebar:
    stock = st.text_input("Stock Symbol", value='SPY')
    start_date = st.date_input("Start Date", value=pd.to_datetime("1995-07-10"))
    end_date = st.date_input("End Date", value=pd.to_datetime("now"))
    training_date = st.date_input("Training End Date", value=pd.to_datetime("2018-01-01"))
    metrics = st.multiselect("Metrics", ['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume'], default=['Adj Close'])
    n_components = st.number_input("Number of Components", min_value=1, max_value=10, value=2)
    buy_state = st.multiselect("Buy States", list(range(10)), default=[1])
    initial_cash = st.number_input("Initial Cash", value=10000.0)
    
# Check if filter values have changed
filters_changed = (
    stock != st.session_state['stock'] or
    start_date != st.session_state['start_date'] or
    end_date != st.session_state['end_date'] or
    training_date != st.session_state['training_date'] or
    metrics != st.session_state['metrics'] or
    n_components != st.session_state['n_components'] or
    buy_state != st.session_state['buy_state'] or
    initial_cash != st.session_state['initial_cash']
)

if filters_changed:
    # Update session state
    st.write('Filters Changed')
    st.session_state['stock'] = stock
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['training_date'] = training_date
    st.session_state['metrics'] = metrics
    st.session_state['n_components'] = n_components
    st.session_state['buy_state'] = buy_state
    st.session_state['initial_cash'] = initial_cash

    # Rerun the script
    st.experimental_rerun()

# Fetch and process data
data_aligned, data = fetch_and_process_data(stock, start_date, end_date, training_date, metrics, n_components)
    
if st.button("Run Analysis"):
    
    st.divider()
    total_return, cash_positions = calculate_hmm_trading_returns(initial_cash, data_aligned.dropna(), buy_state)

    # Buy and Hold Strategy
    max_dd_trading = calculate_max_drawdown(cash_positions)

    buy_and_hold_values = initial_cash / data['Adj Close'].iloc[0] * data['Adj Close']
    buy_and_hold_return = (buy_and_hold_values.iloc[-1] / initial_cash - 1) * 100
    max_dd_buy_hold = calculate_max_drawdown(buy_and_hold_values)


    # Start and end dates for the investment
    start_date = pd.to_datetime(data.index[0])
    end_date = pd.to_datetime(data.index[-1])

    # Calculate CAGR for the trading strategy
    trading_end_value = cash_positions[-1]  # Final value of the portfolio
    cagr_trading = calculate_cagr(initial_cash, trading_end_value, start_date, end_date)

    # Calculate CAGR for the buy-and-hold strategy
    buy_hold_start_value = initial_cash # Initial investment
    buy_hold_end_value = buy_and_hold_values.iloc[-1]  # Final value of the investment
    cagr_buy_hold = calculate_cagr(buy_hold_start_value, buy_hold_end_value, start_date, end_date)

    html_template = """
    <div style="background-color: black; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h2 style="color: blue;">{title}</h2>
        <ul>
            <li><b>Total Return:</b> {total_return:.2f}%</li>
            <li><b>Max Drawdown:</b> {max_dd:.2f}%</li>
            <li><b>CAGR:</b> {cagr:.2f}%</li>
        </ul>
    </div>
    """

    st.markdown(html_template.format(title="Trading Strategy", total_return=total_return, max_dd=max_dd_trading*100, cagr=cagr_trading*100), unsafe_allow_html=True)
    st.markdown(html_template.format(title="Buy and Hold Strategy", total_return=buy_and_hold_return, max_dd=max_dd_buy_hold*100, cagr=cagr_buy_hold*100), unsafe_allow_html=True)





    # Create a figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the Adjusted Close Price line
    fig.add_trace(
        go.Scatter(x=data_aligned.index, y=data_aligned['Adj Close'], line=dict(color='lightgrey'), name='Adjusted Close'),
        secondary_y=False
    )

    # Add markers for each state
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for state in range(n_components):
        state_data = data_aligned[data_aligned['Predicted_State'] == state]
        fig.add_trace(
            go.Scatter(x=state_data.index, y=state_data['Adj Close'], mode='markers', marker=dict(color=colors[state], size=5), name=f'State {state}'),
            secondary_y=False
        )

    # Add a line for Cash/Position Value on secondary y-axis
    fig.add_trace(
        go.Scatter(x=data_aligned.index, y=cash_positions[1:], name='Cash/Position Value', line=dict(color='green')),
        secondary_y=True
    )

    # Update layout for a cleaner look
    fig.update_layout(
        title=f'{stock} Adjusted Close Prices with Market States and Cash/Position Value',
        xaxis_title='Date',
        hovermode='x unified'
    )

    # Set y-axis titles
    fig.update_yaxes(title_text='Adjusted Close Price', secondary_y=False)
    fig.update_yaxes(title_text='Cash/Position Value', secondary_y=True)

    # Show the figure
    st.plotly_chart(fig)



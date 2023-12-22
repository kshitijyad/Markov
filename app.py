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
import seaborn as sns
from PIL import Image
import base64

image_path = "images/kshitij.png"

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
# URL to link to
link_url = "https://www.linkedin.com/in/kshitijyad/"

encoded_image = get_image_base64(image_path)
html_code = f"""
    <a href="{link_url}" target="_blank">
        <img src="data:image/jpeg;base64,{encoded_image}" alt="Clickable Image" style="width:100%">
    </a>
"""

# Display the clickable image in the sidebar
st.sidebar.markdown(html_code, unsafe_allow_html=True)

def get_contrasting_colors(n):
    """Generate 'n' contrasting colors."""
    palette = sns.color_palette("Set3", n)
    return [f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 1)' for rgb in palette]

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
            st.write(f"Metric '{metric}' not found in data.")

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
st.title("Stock Market Analysis Using Hidden Markov Models")

st.markdown('''
### How to Use the Stock Market Analysis App

Welcome to the Stock Market Analysis App, leveraging Hidden Markov Models (HMMs) to provide insights into stock market trends and behaviors. Here's how to get started:

1. **Select Stock and Date Range**: Choose the stock symbol (e.g., 'SPY') and specify the start and end dates for analysis.
2. **Training and Metrics**: Set a training end date for the HMM and select metrics like 'Adjusted Close', 'Volume', etc.
3. **Hidden Markov Model Analysis**: Define the number of hidden states (n_components) in the HMM to analyze market states.
4. **Backtesting Parameters**: Choose 'Buy States' where you'd ideally invest, and input the initial cash for strategy testing.
5. **Monte Carlo Simulation**: Set parameters like window size, time intervals, and iterations for future price simulations.
6. **Run Analysis**: Click "Run Analysis" to view market state predictions, backtesting results, and future price simulations.

Remember, this tool is for informational purposes and not financial advice.

For more on HMMs, visit [Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model).
For financial data analysis, check out [yfinance](https://pypi.org/project/yfinance/).

---
''', unsafe_allow_html=True)

@st.cache_data
def fetch_and_process_data(stock, start_date, end_date, training_date, metrics, n_components):
    data = yf.download(stock, start=start_date, end=end_date, period="1mo")
    if data.empty:
        st.error(f"No data available for {stock} in the specified date range.")
        return None, None
    first_trading_day = data.index.min()
    if pd.to_datetime(training_date) < first_trading_day:
        st.error(f"The first day of trading for {stock} is on {first_trading_day.strftime('%Y-%m-%d')}. "
                 "Please adjust the training date to be on or after this date.")
        return None, None
    #data=data.resample('D').ffill()
    combined_log_returns, training_data_reshaped = calculate_log_returns_and_training_data(data, metrics, training_date)
    combined_log_returns_reshaped = combined_log_returns.values
    if training_data_reshaped.size == 0:
        st.error("Insufficient data for training the model.")
        return data, None
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100000)
    model.fit(np.array(training_data_reshaped).reshape(-1, 1))

    # Predict states for the entire dataset
    predicted_states = model.predict(combined_log_returns_reshaped)
    state_probabilities = model.predict_proba(combined_log_returns_reshaped)
    aligned_dates = data.index[len(data) - len(state_probabilities):]
    latest_date = aligned_dates[-1].strftime('%Y-%m-%d')
    latest_state_probabilities = state_probabilities[-1]
    st.subheader(f"Latest State Probabilities for: {latest_date} ")
    for i, prob in enumerate(latest_state_probabilities):
        st.write(f"State {i} with Probability {np.round(prob,2)*100}% :")
        st.progress(prob)
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
    #colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    colors = get_contrasting_colors(n_components)
    for state in range(n_components):
        state_data = data_aligned[data_aligned['Predicted_State'] == state]
        fig.add_trace(go.Scatter(x=state_data.index, y=state_data['Adj Close'], mode='markers', marker=dict(color=colors[state], size=5), name=f'State {state}'))
    fig.update_layout(title=f'{stock} Adjusted Close Prices with Market States', xaxis_title='Date', yaxis_title='Adjusted Close Price', hovermode='x unified')

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
    return data_aligned,data

def markov_model(data):
    with st.spinner("Running Monte Carlo Simulation..."):
        log_returns = np.log(1 + data[['Adj Close']].pct_change())

        # Running mean and variance (e.g., over the last 60 days)

        running_u = log_returns.rolling(window=window).mean()
        running_var = log_returns.rolling(window=window).var()

        # Parameters for simulation

        # Initialize price list array
        # Extracting the last closing price as a scalar value
        S0 = data['Close'].iloc[-1]
        price_list = np.zeros((t_intervals, iterations))
        price_list[0] = S0

        # Monte Carlo simulation with running mean and variance
        for t in tqdm(range(1, t_intervals)):
            if t < window:
                drift = running_u.iloc[window] - (0.5 * running_var.iloc[window])
                stdev = np.sqrt(running_var.iloc[window])
            else:
                drift = running_u.iloc[t] - (0.5 * running_var.iloc[t])
                stdev = np.sqrt(running_var.iloc[t])
            daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(iterations)))
            price_list[t] = price_list[t - 1] * daily_returns

        confidence_interval = np.percentile(price_list, [10, 50, 90], axis=1)

        # Plot the simulation
        import plotly.graph_objects as go
        last_date = data.index[-1]
        date_range = pd.date_range(last_date, periods=t_intervals, freq='D')
        hovertemplate = 'Date: %{x}<br>Price: %{y:.2f}'
        # Assuming 'price_list' contains all the simulated paths and 'confidence_interval' contains the percentile data

        fig = go.Figure()
        price_paths = price_list.T  
        # Plot each simulated path
        for i, single_path in enumerate(price_paths):
            fig.add_trace(go.Scatter(
                x=date_range,
                y=single_path,
                mode='lines',
                line=dict(width=0.5, color='yellow'),
                opacity=0.5,
                hoverinfo='x+y',
                hovertemplate=hovertemplate,
                showlegend=False
            ))

        # Plot the confidence intervals
        fig.add_trace(go.Scatter(
            x=date_range,
            y=confidence_interval[0],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='10% Percentile',
            hoverinfo='x+y',
            hovertemplate=hovertemplate
        ))
        fig.add_trace(go.Scatter(
            x=date_range,
            y=confidence_interval[1],
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='50% Percentile',
            hoverinfo='x+y',
            hovertemplate=hovertemplate
        ))
        fig.add_trace(go.Scatter(
            x=date_range,
            y=confidence_interval[2],
            mode='lines',
            line=dict(color='green', dash='dash'),
            name='90% Percentile',
            hoverinfo='x+y',
            hovertemplate=hovertemplate
        ))

        # Update the layout of the figure
        fig.update_layout(
        title=f'Simulated {stock} Price Paths with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        template='plotly_white',
        xaxis=dict(
            tickangle=-45,  # Rotate the labels to avoid crowding
            type='date',  # Specify the axis type as date
            dtick=round(t_intervals/10) * 86400000.0,  # Set dtick to a fraction of the total interval, 86400000 is one day in milliseconds
            tickformat='%b %d, %Y',  # Optional: Adjust the date format as needed
            ticklabelmode="period"  # This makes the grid lines and labels align with periods, not the exact dates
        )
    )

        # Use Streamlit's plotly_chart function to display the figure
        # Use Streamlit's pyplot function to display the figure
        st.plotly_chart(fig)

        # Calculate and display Value at Risk (VaR) at 95% confidence level
        VaR_95 = np.percentile(price_list[-1] - S0, 1)
        st.write(f"Value at Risk (95% confidence): {VaR_95}")

        # Calculate and display Expected Shortfall (ES)
        ES = price_list[-1][price_list[-1] - S0 < VaR_95].mean() - S0
        st.write(f"Expected Shortfall: {ES}")
def initialize_state():
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
    if 'window' not in st.session_state:
        st.session_state['window'] = 365
    if 't_intervals' not in st.session_state:
        st.session_state['t_intervals'] = 365
    if 'iterations' not in st.session_state:
        st.session_state['iterations'] = 1000

initialize_state()

with st.sidebar:
    st.sidebar.markdown("## Filter Settings")
    st.sidebar.caption("Enter the ticker symbol of the stock you want to analyze (e.g., 'AAPL' for Apple Inc.).")
    stock = st.sidebar.text_input("Stock Symbol", value='SPY')

    st.sidebar.caption("Select the beginning date for the stock data analysis.")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("1995-07-10"))

    st.sidebar.caption("Choose the ending date up to which you want to analyze the stock data.")
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("now"))

    st.sidebar.caption("Set the cut-off date for training the Hidden Markov Model (HMM). Data up to this date will be used for training.")
    training_date = st.sidebar.date_input("Training End Date", value=pd.to_datetime("2018-01-01"))

    st.sidebar.caption("Choose the stock data metrics to analyze, such as 'Adjusted Close', 'Open', 'High', 'Low', 'Close', and 'Volume'.")
    metrics = st.sidebar.multiselect("Metrics", ['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume'], default=['Adj Close'])

    st.sidebar.caption("Specify the number of hidden states in the HMM. Each state represents a different pattern or regime in the stock's price movement.")
    n_components = st.sidebar.number_input("Number of Components", min_value=1, max_value=10, value=2)

    st.sidebar.markdown("## Backtesting Parameters")
    st.sidebar.caption("Select the states in which you believe it's optimal to buy the stock. These are the states where the market conditions are considered favorable for purchasing.")
    buy_state = st.sidebar.multiselect("Buy States", list(range(10)), default=[1])

    st.sidebar.caption("Input the initial cash amount for backtesting the trading strategy.")
    initial_cash = st.sidebar.number_input("Initial Cash", value=10000.0)
   
    st.sidebar.markdown("## Monte Carlo Simulation")
    st.sidebar.caption("Set the window size for calculating the moving average and variance in the Monte Carlo simulation.")
    window = st.sidebar.number_input("Window", min_value=1, value=150)

    st.sidebar.caption("Choose the number of time intervals for projecting future stock prices in the simulation.")
    t_intervals = st.sidebar.number_input("Time Intervals", min_value=1, value=365)

    st.sidebar.caption("Determine the number of iterations for the Monte Carlo simulation to forecast future stock price movements.")
    iterations = st.sidebar.number_input("Iterations", min_value=1, value=1000)



    
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
    or window != st.session_state['window']
    or t_intervals != st.session_state['t_intervals']
    or iterations != st.session_state['iterations']
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
    st.session_state['window'] = window
    st.session_state['t_intervals'] = t_intervals
    st.session_state['iterations'] = iterations
    # Rerun the script
    st.experimental_rerun()

# Fetch and process data
data_aligned, data = fetch_and_process_data(stock, start_date, end_date, training_date, metrics, n_components)
    
if st.button("Run Analysis"):
    
    st.divider()
    markov_model(data)
    st.divider()
    with st.spinner("Backtesting..."):
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
        #colors = ['blue', 'green', 'red', 'cyan', 'magenta']
        colors = get_contrasting_colors(n_components)

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

disclaimer = """
---
#### Disclaimer
The information provided in this application is for general informational purposes only. It is not intended as financial advice, investment guidance, or a recommendation of any particular strategy or investment. All users should perform their own due diligence and consult with a qualified financial advisor before making any investment decisions. The developers and creators of this application are not responsible for any financial losses or gains incurred as a result of using this application.
"""

st.markdown(disclaimer, unsafe_allow_html=True)

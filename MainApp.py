import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime as datetime
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model

from sklearn.metrics import mean_absolute_error, mean_squared_error


st.set_page_config(page_title="Stock Price Prediction", page_icon=":moneybag:")

# Page title
st.title("Stock Price Visualization")

# Create a form for user inputs
with st.form("user_inputs"):
    # Form fields for collecting user inputs
    start_date = st.date_input("Select a start date", datetime.date.today())
    end_date = st.date_input("Select a end date", datetime.date.today())
    ticker_symbol = st.text_input("Enter the stock ticker symbol (e.g., AAPL)", "AAPL")
    # Form submit button
    submit_button = st.form_submit_button("Generate Plot")
# Check if the user has submitted the inputs
if submit_button:
    data_load_state = st.text("Loading data...")
    # Convert start_date to the desired format
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = datetime.date.today().strftime("%Y-%m-%d") or '2024-01-01'

    df = yf.download(ticker_symbol, start_date, end_date)
    st.write(df.describe())
    data_load_state.text("Data loaded successfully!")

    st.subheader("Closing Price with Date")
    # Plotting the data
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df["Close"])
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Stock Prices for {ticker_symbol} from {start_date} to {end_date}")
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Display the plot
    st.pyplot(fig)

    df.reset_index(inplace=True)
    df.drop(['Date', 'Adj Close'], axis=1, inplace=True)
    

    # Let's understand the 100 days moving average along with 200 days moving average
    ma100 = df['Close'].rolling(100).mean()
    # ma100

    ma200 = df['Close'].rolling(200).mean()
    # ma200


    st.subheader("Closing Price with Date along 100 days moving average")
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df['Close'])
    plt.plot(df['Close'], label='Close Price', color='b')
    plt.plot(ma100, 'r')

    st.pyplot(fig)

    st.subheader("Closing Price with Date along 200 days moving average")
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df['Close'], label='Close Price', color='b')
    plt.plot(ma200, 'g')

    st.pyplot(fig)


    st.write("Now, let's understand the relationship between these two plots")
    
    st.subheader("100 days MA with 200 days MA")
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df['Close'], label='Close Price', color='b')
    plt.plot(ma100, 'r', label='100 days MA')
    plt.plot(ma200, 'g', label='200 days MA')

    st.pyplot(fig)

    st.write("From the above plot, We can say that,\n1. when the 100 days MA, crosses above the 200 days MA, it's a buy signal\n2. When 100 days MA crosses below the 200 days MA, it's a sell signal\n3. When both the MA's are close to each other, it's a neutral signal\n\nThis is a simple moving average strategy to understand the stock price movement\n\nHope you find this helpful!")
    
    prediction_load_state = st.text("\n\nLoading the model for predictions...")

    training_data= pd.DataFrame(df['Close'][0: int(len(df['Close']) * 0.70 )]) # 70% of the data as training data
    test_data= pd.DataFrame(df['Close'][int(len(df['Close']) * 0.70 ): ] )# remaining 30% of the data as testing data
    
    scaler = MinMaxScaler(feature_range=(0,1))
    training_data_array = scaler.fit_transform(training_data)

    # SPlitting the data into x_train and y_train
    x_train = []
    y_train = []

    # Training with 100 days of data, and predicting the 101st day data.
    for i in range(100, training_data_array.shape[0]):
        x_train.append(training_data_array[i-100: i]) # First 0 - 99 days data gone as in training
        y_train.append(training_data_array[i, 0]) # 100th day data as output
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    model = load_model('keras_model1_self.h5')
    
    # To make test predictions for the first row of testing data
    # we need last 100 days data, that we can fetch from the last 100 rows of 
    # training data. 
    past_100_days = training_data.tail(100)

    past_100_days.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    # Concatenate past_100_days and test_data with ordered indices
    final_df_testing = pd.concat([past_100_days, test_data], ignore_index=True)

    testing_data_array = scaler.fit_transform(final_df_testing)

    # Now, create the x_test and y_test, to evaluate the model, and make predictions.
    x_test = []
    y_test = []

    for i in range(100, testing_data_array.shape[0]):
        x_test.append(testing_data_array[i-100: i])
        y_test.append(testing_data_array[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_pred = model.predict(x_test)

    # Now, go inverse scale them, to get actual accuracy and easy analysis of
    # the predictions.

    scale = scaler.scale_[0]
    scale_factor = 1/ scale
    y_test_actual = y_test * scale_factor
    y_pred_actual = y_pred * scale_factor

    pred_fig = plt.figure(figsize=(14, 5))
    plt.plot(y_test, 'g', label='Actual')
    plt.plot(y_pred, 'b', label='Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()
    # plt.show()
    st.pyplot(pred_fig)
    prediction_load_state.text("\n\nPredictions made successfully!")


    metric_show_state = st.text("\n\nCalculating the model metrics...")
    y_pred_train = model.predict(x_train)

    mae = mean_absolute_error(y_train, y_pred_train)
    rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    st.subheader("Model Metrics for Training Data")
    st.write("Mean Absolute Error: ", mae)
    st.write("Root Mean Squared Error: ", rmse)

    st.subheader("Model Metrics for Testing Data")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    st.write("Mean Absolute Error: ", mae)
    st.write("Root Mean Squared Error: ", rmse)
    metric_show_state.text("\n\nModel metrics calculated successfully!")




def main():
    
    
    # TO remove streamlit branding and other running animation
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Spinners
    bar = st.progress(0)
    for i in range(101):
        bar.progress(i)
        # time.sleep(0.02)  # Adjust the sleep time for the desired speed

    st.balloons()

    # Web content starts
    # Navbar starts
        # Create the Streamlit app
    col1, col2 = st.columns([1, 10])
    with col1:
        st.header("	:moneybag:")
    with col2:
        st.header("Stock Market Trend predictor")
        

    

    # Create sidebar section for app description and links
    # Sidebar content
    st.sidebar.header("Stock Market Trend Predictor :moneybag:")
    st.sidebar.write("Description :male-detective:")
    st.sidebar.write("""
    This web app visualizes historical stock prices and predicts future trends using deep learning techniques. It analyzes stock prices over a specified period, calculates moving averages, and uses a Long Short-Term Memory (LSTM) neural network to predict future price movements.
        
    \nSkills Enhanced:\n
    - 📈 Time Series Analysis
    - 💻 Deep Learning
    - 🐍 Python
    - 📊 Data Visualization

    \nSteps:\n
    1. Data Acquisition: Fetch historical stock prices using the Yahoo Finance API.
    2. Data Preprocessing: Clean data, calculate moving averages, and prepare training/testing sets.
    3. Data Visualization: Visualize stock prices, moving averages, and predictive model results.
    4. Model Training: Train an LSTM neural network on historical stock prices.
    5. Model Prediction: Predict future stock prices using the trained model.

    By leveraging deep learning, this app helps users understand stock price trends and make informed investment decisions.
            
    \n**Credits** 🌟\n
    Developer: Aniket Panchal
    GitHub: [Aniket2021448](https://github.com/Aniket2021448)

    \n**Contact** 📧\n
    For inquiries or feedback, please contact aniketpanchal1257@gmail.com
    **Portfolio** 💼\n
    \nCheck out my portfolio website: [Your Portfolio Website Link](https://aniket2021448.github.io/My-portfolio/)
    """)
    
    # Dropdown menu for other app links

    st.sidebar.write("In case the apps are down, because of less usage")
    st.sidebar.write("Kindly reach out to me @ aniketpanchal1257@gmail.com")
    

    # Create the form
    

if __name__ == "__main__":
    main()

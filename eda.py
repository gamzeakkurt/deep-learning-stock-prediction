import seaborn as sns
import matplotlib.pyplot as plt
import quantstats as qs
import pandas as pd
import numpy as np

def perform_eda(df, tickers, companies):
    """
    Perform exploratory data analysis (EDA) on stock market data.
    """

    # -------------------------------
    # 📌 Distribution of Open Prices
    # -------------------------------

    sns.set(rc={'figure.figsize': (11.7, 8.27)}, palette="pastel")  # Soft color palette

    plt.figure(figsize=(11.7, 8.27))

    for i, (ticker, company) in enumerate(zip(tickers, companies), 1):
        ticker_data = df[df['Ticker'] == ticker]
        skew_value = qs.stats.skew(ticker_data['Open']).round(3)

        plt.subplot(2, 2, i)
        sns.histplot(data=ticker_data, x='Open', kde=True, color=sns.color_palette()[i - 1])
        
        plt.title(f"Histogram of 'Open' Prices for {company}")
        plt.xlabel("Open Price")
        plt.ylabel("Frequency")

        plt.text(0.05, -0.35, f"Skewness: {skew_value}", fontsize=10, ha='left', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()


    # -------------------------------
    # 📌 Distribution of Close Prices
    # -------------------------------

    sns.set(rc={'figure.figsize': (11.7, 8.27)}, palette="Reds")

    plt.figure(figsize=(11.7, 8.27))

    for i, (ticker, company) in enumerate(zip(tickers, companies), 1):
        ticker_data = df[df['Ticker'] == ticker]
        skew_value = qs.stats.skew(ticker_data['Close']).round(3)

        plt.subplot(2, 2, i)
        sns.histplot(data=ticker_data, x='Close', kde=True, color=sns.color_palette("Reds")[i + 1])

        plt.title(f"Histogram of 'Close' Prices for {company}")
        plt.xlabel("Close Price")
        plt.ylabel("Frequency")

        plt.text(0.05, -0.35, f"Skewness: {skew_value}", fontsize=10, ha='left', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()


    # -------------------------------
    # 📌 Open vs. Close Price Scatter Plot
    # -------------------------------

    sns.set(rc={'figure.figsize': (11.7, 8.27)}, palette="pastel")

    plt.figure(figsize=(11.7, 8.27))

    for i, (ticker, company) in enumerate(zip(tickers, companies), 1):
        ticker_data = df[df['Ticker'] == ticker]

        plt.subplot(2, 2, i)
        sns.scatterplot(x='Open', y='Close', data=ticker_data, color=sns.color_palette()[i - 1])

        plt.title(f"{company} - Open vs Close Price")
        plt.xlabel("Open Price")
        plt.ylabel("Close Price")

    plt.tight_layout()
    plt.show()


    # -------------------------------
    # 📌 Daily Returns Calculation
    # -------------------------------

    df['Daily Return'] = df.groupby('Ticker')['Adj Close'].pct_change()

    # Extract daily returns for each stock
    returns_data = {ticker: df[df['Ticker'] == ticker]['Daily Return'].dropna() for ticker in tickers}

    # -------------------------------
    # 📌 Daily Return Time Series Plots
    # -------------------------------

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    for i, (ticker, ax) in enumerate(zip(tickers, axes.flat)):
        returns_data[ticker].plot(ax=ax, legend=True, linestyle='--', marker='o')
        ax.set_title(ticker.upper())

    fig.tight_layout()
    plt.show()


    # -------------------------------
    # 📌 Histogram of Daily Returns
    # -------------------------------
    companies = {
        'AAPL': df[df['Ticker'] == 'AAPL'],
        'TSLA': df[df['Ticker'] == 'TSLA'],
        'MSFT': df[df['Ticker'] == 'MSFT'],
        'AMZN': df[df['Ticker'] == 'AMZN']
    }

    aapl_returns = df[df['Ticker'] == 'AAPL']['Daily Return'].dropna()
    tsla_returns = df[df['Ticker'] == 'TSLA']['Daily Return'].dropna()
    msft_returns= df[df['Ticker'] == 'MSFT']['Daily Return'].dropna()
    amzn_returns=df[df['Ticker'] == 'AMZN']['Daily Return'].dropna()

    #plt.figure(figsize=(15, 10))
    #print('\nApple Daily Returns Histogram')
    #qs.plots.histogram(aapl_returns, resample='D')
    #plt.show()




    # -------------------------------
    # 📌 Adjusted Closing Price Plot
    # -------------------------------

    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=0.9, bottom=0.1)

    for i, (ticker, company_data) in enumerate(companies.items(), 1):
        plt.subplot(2, 2, i)
        company_data['Adj Close'].plot(color='red', legend=False)
        plt.ylabel('Adj Close')
        plt.title(f"Adjusted Closing Price of {ticker}")

    plt.tight_layout()
    plt.show()


    # -------------------------------
    # 📌 Sales Volume Plot
    # -------------------------------

    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=0.9, bottom=0.1)

    for i, (ticker, company) in enumerate(companies.items(), 1):
        plt.subplot(2, 2, i)
        company['Volume'].plot(color='red', legend=False)
        plt.ylabel('Volume')
        plt.title(f"Sales Volume for {ticker}")

    plt.tight_layout()
    plt.show()


    # -------------------------------
    # 📌 Moving Averages Calculation
    # -------------------------------

    ma_days = [10, 20, 50]

    for ma in ma_days:
        for ticker, company in companies.items():
            company[f"MA {ma} days"] = company['Adj Close'].rolling(ma).mean()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    for i, (ticker, ax) in enumerate(zip(tickers, axes.flat)):
        companies[ticker][['Adj Close'] + [f"MA {ma} days" for ma in ma_days]].plot(ax=ax)
        ax.set_title(ticker.upper())

    fig.tight_layout()
    plt.show()


    # -------------------------------
    # 📌 Correlation of Closing Prices
    # -------------------------------

    adj_close_df = df.pivot_table(values='Daily Return', index='Date', columns='Ticker')

    sns.jointplot(x='AAPL', y='MSFT', data=adj_close_df, kind='scatter', color='seagreen')

    """
    - The joint plot illustrates the relationship between Apple (AAPL) and Microsoft (MSFT) daily returns.
    - The scatter plot at the center suggests a positive correlation.
    - Histograms on the top and right show the individual return distributions.
    - The spread suggests some degree of volatility, but the positive trend implies these stocks move in sync.
    """


    # -------------------------------
    # 📌 PairGrid for Stock Correlation
    # -------------------------------

    returns_fig = sns.PairGrid(adj_close_df)

    returns_fig.map_upper(plt.scatter, color='red')
    returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
    returns_fig.map_diag(plt.hist, bins=30)


    # -------------------------------
    # 📌 Heatmaps of Correlations
    # -------------------------------

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    sns.heatmap(adj_close_df.corr(), annot=True, cmap='PuBu', linewidths=0.3, square=True, cbar_kws={"shrink": 0.8})
    plt.title('Correlation of Stock Returns', fontsize=14)

    plt.subplot(2, 2, 2)
    sns.heatmap(adj_close_df.corr(), annot=True, cmap='BuGn', linewidths=0.3, square=True, cbar_kws={"shrink": 0.8})
    plt.title('Correlation of Stock Closing Prices', fontsize=14)

    plt.show()


    # -------------------------------
    # 📌 Risk vs. Expected Return
    # -------------------------------

    rets = adj_close_df.dropna()
    area = np.pi * 20

    plt.figure(figsize=(10, 8))
    plt.scatter(rets.mean(), rets.std(), s=area)
    plt.xlabel('Expected Return')
    plt.ylabel('Risk')

    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points',
                     ha='right', va='bottom', arrowprops=dict(arrowstyle='-', color='red', connectionstyle='arc3,rad=-0.3'))

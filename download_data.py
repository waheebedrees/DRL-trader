# save this as download_data.py — downloads and saves to CSV
"""
Download market data from Yahoo Finance and save to CSV.
Run once, then use the CSV files for training.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


def download_and_save(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    save_dir: str = ".",
) -> str:
    """
    Download data and save to CSV in the specified directory.
    
    Returns the filepath of the saved CSV.
    """
    print(f"Downloading {symbol} {interval} from {start} to {end}...")
    
    ticker = yf.Ticker(symbol)
    
    # For hourly data, Yahoo limits to last 730 days
    if interval in ("1h", "60m") and "2023" in start:
        # Try the requested range first
        df = ticker.history(start=start, end=end, interval=interval)
        
        if df.empty:
            # Fall back to last 730 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=729)
            print(f"  Hourly data limited to last 730 days, using:")
            print(f"    {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
            )
    else:
        df = ticker.history(start=start, end=end, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    
    # Standardize columns
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume',
    })
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    df = df[df['close'] > 0]
    df = df[df['volume'] > 0]
    
    # Create filename
    # e.g., BTC-USD_1h_2024-05-31_to_2026-05-30.csv
    date_from = df.index[0].strftime('%Y-%m-%d')
    date_to = df.index[-1].strftime('%Y-%m-%d')
    filename = f"{symbol}_{interval}_{date_from}_to_{date_to}.csv"
    filepath = os.path.join(save_dir, filename)
    
    # Save with timestamp index
    df.to_csv(filepath, index=True, index_label='timestamp')
    
    print(f"  ✓ Saved {len(df):,} bars to: {filepath}")
    print(f"  Date range: {date_from} to {date_to}")
    print(f"  Price: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print()
    
    return filepath


if __name__ == "__main__":
    # Create a data directory if it doesn't exist
    data_dir = "market_data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print(" Downloading Market Data")
    print("=" * 60)
    print(f" Saving to: {os.path.abspath(data_dir)}/")
    print()
    
    files = []
    
    # 1. Bitcoin hourly (2 years of data for intraday trading)
    files.append(download_and_save(
        symbol="BTC-USD",
        start="2023-12-01",
        end="2025-06-01",
        interval="1h",
        save_dir=data_dir,
    ))
    
    # 2. Bitcoin daily (longer history for swing trading)
    files.append(download_and_save(
        symbol="BTC-USD",
        start="2018-01-01",
        end="2025-06-01",
        interval="1d",
        save_dir=data_dir,
    ))
    
    # 3. S&P 500 ETF daily (for stock market comparison)
    files.append(download_and_save(
        symbol="SPY",
        start="2018-01-01",
        end="2025-06-01",
        interval="1d",
        save_dir=data_dir,
    ))
    
    # 4. Apple daily
    files.append(download_and_save(
        symbol="AAPL",
        start="2018-01-01",
        end="2025-06-01",
        interval="1d",
        save_dir=data_dir,
    ))
    
    # 5. Ethereum hourly
    files.append(download_and_save(
        symbol="ETH-USD",
        start="2023-12-01",
        end="2025-06-01",
        interval="1h",
        save_dir=data_dir,
    ))
    
    print("=" * 60)
    print(" ✓ All downloads complete!")
    print("=" * 60)
    print()
    print("Files saved:")
    for f in files:
        print(f"  • {f}")
    print()
    print("Next step: Run train_real_data.py with your chosen CSV file")
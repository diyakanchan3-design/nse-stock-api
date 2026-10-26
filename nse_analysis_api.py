"""
NSE Stock Analysis API  (version 2)
===================================
One FastAPI service that does all the data work for the NSE stock app.

ORIGINAL ENDPOINTS
  /analyze/{symbol}          full technical + fundamental analysis
  /chart/{symbol}            candlestick + indicator history
  /watchlist?symbols=...     quick view of many stocks

NEW ENDPOINTS (all free, no new libraries)
  /symbols                   list of big NSE stocks with sectors
  /compare?symbols=A,B       compare 2-4 stocks (starts at 100) + correlation
  /risk/{symbol}             volatility, drawdown, Sharpe, beta, VaR
  /signal/{symbol}           simple Bullish / Neutral / Bearish score
  /backtest/{symbol}         moving average strategy vs buy-and-hold
  /market-overview           NIFTY, SENSEX, BANK NIFTY, NIFTY IT, India VIX
  /movers                    top gainers, top losers, sector performance
  /portfolio?holdings=...    profit/loss tracker

REQUIREMENTS:  pip install fastapi uvicorn yfinance pandas numpy
RUN LOCALLY:   uvicorn nse_analysis_api:app --reload --port 8000
TEST PAGE:     http://127.0.0.1:8000/docs   (free, built in)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import math

app = FastAPI(title="NSE Stock Analysis API", version="2.0")

# Allow any frontend (v0.dev, Lovable, or anywhere else) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── INDICATOR FUNCTIONS (same ones built in our Jupyter session today) ────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_moving_averages(prices, short_window=50, long_window=200):
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    return ma_short, ma_long

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100

def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth — the corrected version, fixing the bug we caught earlier today where fixed-position matching broke on stocks with an irregularly reported quarter."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def safe_round(val, decimals=2):
    if val is None or pd.isna(val):
        return None
    return round(float(val), decimals)


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    """Same date-matched logic as earnings growth, applied to revenue."""
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Extended fundamentals: earnings growth, revenue growth, margins, ROE, debt-to-equity. IMPORTANT: this deliberately avoids stock.info — Yahoo Finance blocks/rate-limits that specific endpoint for many cloud hosting providers (Render, Railway, Heroku), returning an almost-empty dict even though the price-history endpoint works fine. Instead, everything here is calculated directly from the income statement and balance sheet, which use the endpoint that's actually working for us."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        # margins calculated directly — same math a real analyst would do by hand
        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                # ROE using latest quarter's net income annualized (x4) against latest equity —
                # a simplified TTM-style approximation, standard when only one quarter is available
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    # market cap: try the lighter fast_info endpoint, which is less likely to be blocked
    # than the full .info dict — separate try block so a failure here doesn't affect anything above
    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass  # market cap is a nice-to-have; not worth failing the request over

    return fundamentals


def get_technical_snapshot(hist):
    """The core technical calculation, factored out so both /analyze and /watchlist can reuse it."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "NSE Stock Analysis API v2 is running",
        "endpoints": ["/analyze/{symbol}", "/chart/{symbol}", "/watchlist", "/symbols", "/compare", "/risk/{symbol}", "/signal/{symbol}", "/backtest/{symbol}", "/market-overview", "/movers", "/portfolio", "/docs"],
    }


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_fundamentals(stock)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "fundamentals": fundamentals,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """ OHLC candlestick data + overlay indicator series for charting. Returns one row per trading day: date, OHLCV, RSI, MACD line/signal, MA50, MA200. `months` controls how much history to return in the chart (default 6, max 24). """
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    months = min(max(months, 1), 24)

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)  # need 2y so MA200 has enough lookback, even though we only return the recent slice
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        trading_days = months * 21  # approx trading days per month
        recent = hist.tail(trading_days)

        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]) if date in rsi.index else None,
                "macd_line": safe_round(macd_line.loc[date]) if date in macd_line.index else None,
                "macd_signal": safe_round(signal_line.loc[date]) if date in signal_line.index else None,
                "ma_50": safe_round(ma50.loc[date]) if date in ma50.index else None,
                "ma_200": safe_round(ma200.loc[date]) if date in ma200.index else None,
            })

        return {"symbol": ticker, "months": months, "data": rows}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """ Lightweight multi-stock snapshot for a watchlist view. Pass symbols comma-separated: /watchlist?symbols=RELIANCE,TCS,INFY Capped at 10 symbols per request to keep response time reasonable on free hosting. """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", auto_adjust=True)
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue

            technical, corporate_action_flag = get_technical_snapshot(hist)
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return {"count": len(results), "stocks": results}


# ═══════════════════════════════════════════════════════════════════════════
#  NEW FEATURES (version 2)
#  All free. All use only yfinance + pandas + numpy (same as before),
#  so requirements.txt does NOT change.
# ═══════════════════════════════════════════════════════════════════════════

RISK_FREE_RATE = 0.07      # about the Indian government bond yield; used for the Sharpe ratio
CACHE_SECONDS = 300        # keep data for 5 minutes so Yahoo is not called again and again
_cache = {}

INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "INDIA VIX": "^INDIAVIX",
}

# A basket of big NSE companies with their sector. You can add or remove names here.
LARGE_CAPS = {
    "RELIANCE": "Energy", "ONGC": "Energy", "COALINDIA": "Energy",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking",
    "BAJFINANCE": "Financial Services", "BAJAJFINSV": "Financial Services",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "MARUTI": "Auto", "M&M": "Auto", "EICHERMOT": "Auto",
    "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma",
    "LT": "Infrastructure", "ULTRACEMCO": "Cement",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "NTPC": "Power", "POWERGRID": "Power",
    "BHARTIARTL": "Telecom", "TITAN": "Consumer", "ASIANPAINT": "Consumer",
}


def cache_get(key):
    item = _cache.get(key)
    if item and time.time() - item[0] < CACHE_SECONDS:
        return item[1]
    return None


def cache_set(key, value):
    _cache[key] = (time.time(), value)
    return value


def to_ticker(symbol):
    symbol = symbol.upper().strip()
    if symbol.startswith("^") or symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


def get_history(ticker, period="2y"):
    key = ("hist", ticker, period)
    hit = cache_get(key)
    if hit is not None:
        return hit
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if hist is not None and not hist.empty:
        cache_set(key, hist)
    return hist


def get_close(ticker, period="2y"):
    """Closing prices with a clean date index (no timezone), so two stocks line up easily."""
    hist = get_history(ticker, period)
    if hist is None or hist.empty:
        return None
    close = hist["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close


def downsample(items, step):
    """Keep every Nth row (and always the last one) so charts load fast."""
    if len(items) <= 1:
        return items
    kept = items[::step]
    if kept[-1] is not items[-1]:
        kept.append(items[-1])
    return kept


# ── FEATURE 1: List of symbols (for the search box / dropdown) ─────────────

@app.get("/symbols")
def list_symbols():
    """Big NSE companies with their sector. Use it to fill a dropdown in the app."""
    return {
        "count": len(LARGE_CAPS),
        "symbols": [{"symbol": s, "sector": sec} for s, sec in LARGE_CAPS.items()],
    }


# ── FEATURE 2: Compare stocks ──────────────────────────────────────────────

@app.get("/compare")
def compare_stocks(symbols: str, months: int = 12):
    """
    Compare 2 to 4 stocks. All prices start at 100 so you can see who grew more.
    Example: /compare?symbols=TCS,INFY,WIPRO&months=12
    """
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()][:4]
    if len(symbol_list) < 2:
        raise HTTPException(status_code=400, detail="Give at least 2 symbols, e.g. ?symbols=TCS,INFY")
    months = min(max(months, 1), 24)

    closes, missing = {}, []
    for s in symbol_list:
        ticker = to_ticker(s)
        try:
            close = get_close(ticker, "2y")
        except Exception:
            close = None
        if close is None:
            missing.append(ticker)
        else:
            closes[ticker] = close

    if len(closes) < 2:
        raise HTTPException(status_code=404, detail=f"Not enough valid symbols. No data for: {missing}")

    df = pd.DataFrame(closes).dropna().tail(months * 21)
    if len(df) < 2:
        raise HTTPException(status_code=404, detail="Not enough common trading days to compare.")

    rebased = df / df.iloc[0] * 100
    total_return = (df.iloc[-1] / df.iloc[0] - 1) * 100
    corr = df.pct_change().dropna().corr()

    series = []
    for date, row in rebased.iterrows():
        point = {"date": date.strftime("%Y-%m-%d")}
        for t in rebased.columns:
            point[t] = safe_round(row[t])
        series.append(point)

    pairs = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = safe_round(corr.loc[cols[i], cols[j]], 2)
            if c is None:
                meaning = "Not enough data"
            elif c >= 0.7:
                meaning = "Move together a lot - holding both gives little diversification"
            elif c >= 0.3:
                meaning = "Somewhat linked - some diversification benefit"
            else:
                meaning = "Move quite independently - good for diversification"
            pairs.append({"pair": f"{cols[i]} vs {cols[j]}", "correlation": c, "meaning": meaning})

    returns_dict = {t: safe_round(total_return[t]) for t in df.columns}
    best = max(returns_dict, key=lambda k: returns_dict[k])
    worst = min(returns_dict, key=lambda k: returns_dict[k])

    return {
        "months": months,
        "symbols": list(df.columns),
        "missing": missing,
        "total_return_pct": returns_dict,
        "best_performer": best,
        "worst_performer": worst,
        "correlations": pairs,
        "series": downsample(series, 2),
    }


# ── FEATURE 3: Risk report ─────────────────────────────────────────────────

@app.get("/risk/{symbol}")
def risk_metrics(symbol: str, years: int = 1):
    """Volatility, max drawdown, Sharpe ratio, Beta (vs NIFTY 50) and Value at Risk."""
    ticker = to_ticker(symbol)
    years = min(max(years, 1), 2)

    try:
        close = get_close(ticker, "2y")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching {ticker}: {e}")
    if close is None:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

    close = close.tail(252 * years + 1)
    returns = close.pct_change().dropna()
    if len(returns) < 30:
        raise HTTPException(status_code=404, detail="Not enough price history to measure risk.")

    ann_vol = returns.std() * math.sqrt(252)
    cagr = (close.iloc[-1] / close.iloc[0]) ** (252 / len(returns)) - 1
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else None

    drawdown = close / close.cummax() - 1
    max_dd = drawdown.min() * 100
    max_dd_date = drawdown.idxmin().strftime("%Y-%m-%d")
    var_95 = -np.percentile(returns, 5) * 100

    beta = None
    try:
        index_close = get_close("^NSEI", "2y")
        if index_close is not None:
            joined = pd.concat([returns, index_close.pct_change()], axis=1, join="inner").dropna()
            joined.columns = ["stock", "market"]
            if len(joined) > 30 and joined["market"].var() > 0:
                beta = joined["stock"].cov(joined["market"]) / joined["market"].var()
    except Exception:
        beta = None  # beta is a nice-to-have; do not fail the whole request

    vol_pct = ann_vol * 100
    if vol_pct < 20:
        risk_level = "Low"
    elif vol_pct < 35:
        risk_level = "Medium"
    else:
        risk_level = "High"

    if beta is None:
        beta_note = "Beta could not be calculated."
    elif beta > 1.2:
        beta_note = "Moves more than the market - gains and losses are both bigger than NIFTY."
    elif beta < 0.8:
        beta_note = "Moves less than the market - calmer than NIFTY."
    else:
        beta_note = "Moves roughly in line with the market."

    dd_series = [
        {"date": d.strftime("%Y-%m-%d"), "drawdown_pct": safe_round(v * 100)}
        for d, v in drawdown.items()
    ]

    return {
        "symbol": ticker,
        "years_used": years,
        "risk_level": risk_level,
        "annual_volatility_pct": safe_round(vol_pct),
        "annual_return_pct": safe_round(cagr * 100),
        "sharpe_ratio": safe_round(sharpe),
        "beta_vs_nifty": safe_round(beta),
        "beta_note": beta_note,
        "max_drawdown_pct": safe_round(max_dd),
        "max_drawdown_date": max_dd_date,
        "value_at_risk_95_pct": safe_round(var_95),
        "var_note": f"On a normal bad day (1 in 20), the stock can fall about {safe_round(var_95)}% or more.",
        "best_day_pct": safe_round(returns.max() * 100),
        "worst_day_pct": safe_round(returns.min() * 100),
        "risk_free_rate_used_pct": round(RISK_FREE_RATE * 100, 2),
        "drawdown_series": downsample(dd_series, 3),
    }


# ── FEATURE 4: Buy / Hold / Sell signal ────────────────────────────────────

@app.get("/signal/{symbol}")
def trading_signal(symbol: str):
    """
    A simple rule-based score from -5 to +5, built from the indicators we already calculate.
    For learning only - not investment advice.
    """
    ticker = to_ticker(symbol)
    try:
        hist = get_history(ticker, "2y")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching {ticker}: {e}")
    if hist is None or hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

    tech, corporate_flag = get_technical_snapshot(hist)
    score, reasons = 0, []

    rsi = tech["rsi_14"]
    if rsi is not None:
        if rsi < 30:
            score += 1; reasons.append({"indicator": "RSI", "points": 1, "why": f"RSI is {rsi} - below 30, so the stock looks oversold (may bounce)."})
        elif rsi > 70:
            score -= 1; reasons.append({"indicator": "RSI", "points": -1, "why": f"RSI is {rsi} - above 70, so the stock looks overbought (may cool down)."})
        else:
            reasons.append({"indicator": "RSI", "points": 0, "why": f"RSI is {rsi} - in the normal zone."})

    hist_val = tech["macd_histogram"]
    if hist_val is not None:
        if hist_val > 0:
            score += 1; reasons.append({"indicator": "MACD", "points": 1, "why": "MACD is above its signal line - momentum is positive."})
        else:
            score -= 1; reasons.append({"indicator": "MACD", "points": -1, "why": "MACD is below its signal line - momentum is weak."})

    if tech["ma_trend"] == "bullish":
        score += 1; reasons.append({"indicator": "50/200 Moving Average", "points": 1, "why": "The 50-day average is above the 200-day average - long-term trend is up."})
    elif tech["ma_trend"] == "bearish":
        score -= 1; reasons.append({"indicator": "50/200 Moving Average", "points": -1, "why": "The 50-day average is below the 200-day average - long-term trend is down."})

    price, ma50 = tech["current_price"], tech["ma_50"]
    if price is not None and ma50 is not None:
        if price > ma50:
            score += 1; reasons.append({"indicator": "Price vs 50-day average", "points": 1, "why": "Price is above its 50-day average - short-term trend is up."})
        else:
            score -= 1; reasons.append({"indicator": "Price vs 50-day average", "points": -1, "why": "Price is below its 50-day average - short-term trend is down."})

    pb = tech["bollinger_percent_b"]
    if pb is not None:
        if pb < 0.2:
            score += 1; reasons.append({"indicator": "Bollinger Bands", "points": 1, "why": "Price is near the lower band - it is cheap compared with its recent range."})
        elif pb > 0.8:
            score -= 1; reasons.append({"indicator": "Bollinger Bands", "points": -1, "why": "Price is near the upper band - it is expensive compared with its recent range."})
        else:
            reasons.append({"indicator": "Bollinger Bands", "points": 0, "why": "Price is in the middle of its recent range."})

    adx = tech["adx_14"]
    trend_strength = None
    if adx is not None:
        trend_strength = "Strong trend" if adx >= 25 else "Weak or no trend"

    if score >= 3:
        verdict = "Strong Bullish"
    elif score >= 1:
        verdict = "Mildly Bullish"
    elif score == 0:
        verdict = "Neutral"
    elif score >= -2:
        verdict = "Mildly Bearish"
    else:
        verdict = "Strong Bearish"

    return {
        "symbol": ticker,
        "score": score,
        "score_range": [-5, 5],
        "verdict": verdict,
        "trend_strength": trend_strength,
        "reasons": reasons,
        "corporate_action_flag": corporate_flag,
        "disclaimer": "This is a simple rule-based score for learning. It is not investment advice.",
    }


# ── FEATURE 5: Strategy backtest (moving average crossover) ────────────────

@app.get("/backtest/{symbol}")
def backtest(symbol: str, short: int = 50, long: int = 200, years: int = 3):
    """
    Test a simple strategy on past prices: BUY when the short average crosses above
    the long average, SELL (go to cash) when it crosses below.
    Compares it with just buying and holding. Costs and taxes are not included.
    """
    ticker = to_ticker(symbol)
    if short >= long:
        raise HTTPException(status_code=400, detail="'short' must be smaller than 'long', e.g. short=50&long=200")
    if short < 2 or long > 250:
        raise HTTPException(status_code=400, detail="Use short >= 2 and long <= 250.")
    years = min(max(years, 1), 5)

    try:
        close = get_close(ticker, "5y")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching {ticker}: {e}")
    if close is None:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

    ma_short = close.rolling(short).mean()
    ma_long = close.rolling(long).mean()
    df = pd.DataFrame({"close": close, "ma_short": ma_short, "ma_long": ma_long}).dropna()
    df = df.tail(252 * years)
    if len(df) < 60:
        raise HTTPException(status_code=404, detail="Not enough history for this test. Try a shorter 'long' or fewer years.")

    position = (df["ma_short"] > df["ma_long"]).astype(int)
    daily_ret = df["close"].pct_change().fillna(0)
    strat_ret = position.shift(1).fillna(0) * daily_ret   # act on the NEXT day, so no cheating with future data

    strat_curve = (1 + strat_ret).cumprod() * 100
    hold_curve = (1 + daily_ret).cumprod() * 100

    def max_drawdown(curve):
        return safe_round(((curve / curve.cummax()) - 1).min() * 100)

    # list of finished trades
    trades, in_trade, entry = [], False, None
    for price, pos in zip(df["close"].values, position.values):
        if pos == 1 and not in_trade:
            in_trade, entry = True, price
        elif pos == 0 and in_trade:
            trades.append(price / entry - 1)
            in_trade = False
    wins = sum(1 for t in trades if t > 0)

    curve_rows = [
        {"date": d.strftime("%Y-%m-%d"), "strategy": safe_round(strat_curve.loc[d]), "buy_and_hold": safe_round(hold_curve.loc[d])}
        for d in df.index
    ]

    strat_total = safe_round(strat_curve.iloc[-1] - 100)
    hold_total = safe_round(hold_curve.iloc[-1] - 100)
    winner = "Strategy" if strat_total > hold_total else "Buy and hold"

    return {
        "symbol": ticker,
        "rule": f"Buy when {short}-day average is above {long}-day average, otherwise stay in cash",
        "years_tested": years,
        "strategy_return_pct": strat_total,
        "buy_and_hold_return_pct": hold_total,
        "winner": winner,
        "strategy_max_drawdown_pct": max_drawdown(strat_curve),
        "buy_and_hold_max_drawdown_pct": max_drawdown(hold_curve),
        "completed_trades": len(trades),
        "winning_trades": wins,
        "win_rate_pct": safe_round(wins / len(trades) * 100) if trades else None,
        "currently_in_market": bool(position.iloc[-1] == 1),
        "note": "Past results do not guarantee future results. Brokerage, tax and slippage are not included.",
        "curve": downsample(curve_rows, 5),
    }


# ── FEATURE 6: Market overview (indices) ───────────────────────────────────

@app.get("/market-overview")
def market_overview():
    """Latest level and change for NIFTY 50, SENSEX, BANK NIFTY, NIFTY IT and India VIX."""
    results = []
    for name, ticker in INDICES.items():
        try:
            close = get_close(ticker, "3mo")
            if close is None or len(close) < 2:
                results.append({"name": name, "symbol": ticker, "error": "No data"})
                continue
            last, prev = float(close.iloc[-1]), float(close.iloc[-2])
            month_base = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
            results.append({
                "name": name,
                "symbol": ticker,
                "level": safe_round(last),
                "change": safe_round(last - prev),
                "change_pct": safe_round((last / prev - 1) * 100),
                "one_month_change_pct": safe_round((last / month_base - 1) * 100),
                "as_of": close.index[-1].strftime("%Y-%m-%d"),
            })
        except Exception as e:
            results.append({"name": name, "symbol": ticker, "error": str(e)})

    return {"timestamp": datetime.now().isoformat(), "indices": results}


# ── FEATURE 7: Top gainers, top losers and sector performance ─────────────

def get_large_cap_closes():
    key = ("largecaps",)
    hit = cache_get(key)
    if hit is not None:
        return hit
    tickers = [f"{s}.NS" for s in LARGE_CAPS]
    data = yf.download(tickers, period="5d", auto_adjust=True, progress=False, threads=True)
    return cache_set(key, data["Close"])


@app.get("/movers")
def movers(top: int = 5):
    """Today's top gainers and losers, plus average change by sector, from a basket of big NSE stocks."""
    top = min(max(top, 1), 10)
    try:
        closes = get_large_cap_closes()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch market data right now: {e}")

    rows = []
    for t in closes.columns:
        s = closes[t].dropna()
        if len(s) < 2:
            continue
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        sym = str(t).replace(".NS", "")
        rows.append({
            "symbol": sym,
            "sector": LARGE_CAPS.get(sym, "Other"),
            "price": safe_round(last),
            "change_pct": safe_round((last / prev - 1) * 100),
        })

    if not rows:
        raise HTTPException(status_code=502, detail="No price data came back. Try again in a minute.")

    rows.sort(key=lambda r: r["change_pct"], reverse=True)

    sector_totals = {}
    for r in rows:
        sector_totals.setdefault(r["sector"], []).append(r["change_pct"])
    sectors = [
        {"sector": sec, "avg_change_pct": safe_round(sum(v) / len(v)), "stocks": len(v)}
        for sec, v in sector_totals.items()
    ]
    sectors.sort(key=lambda x: x["avg_change_pct"], reverse=True)

    return {
        "timestamp": datetime.now().isoformat(),
        "stocks_covered": len(rows),
        "top_gainers": rows[:top],
        "top_losers": list(reversed(rows[-top:])),
        "sectors": sectors,
        "all_stocks": rows,
    }


# ── FEATURE 8: Portfolio tracker ───────────────────────────────────────────

@app.get("/portfolio")
def portfolio(holdings: str):
    """
    Profit/loss for a small portfolio. Format: SYMBOL:QUANTITY:BUY_PRICE, comma separated.
    Example: /portfolio?holdings=RELIANCE:10:2450,TCS:5:3800,INFY:20:1500
    """
    positions = []
    for part in holdings.split(",")[:15]:
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            raise HTTPException(status_code=400, detail=f"'{part}' is not valid. Use SYMBOL:QUANTITY:BUY_PRICE")
        try:
            qty, buy = float(bits[1]), float(bits[2])
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Quantity and buy price must be numbers in '{part}'")
        if qty <= 0 or buy <= 0:
            raise HTTPException(status_code=400, detail=f"Quantity and buy price must be above 0 in '{part}'")
        positions.append((bits[0].strip().upper(), qty, buy))

    if not positions:
        raise HTTPException(status_code=400, detail="Give at least one holding, e.g. ?holdings=RELIANCE:10:2450")

    rows, not_found = [], []
    for sym, qty, buy in positions:
        ticker = to_ticker(sym)
        try:
            close = get_close(ticker, "1mo")
        except Exception:
            close = None
        if close is None:
            not_found.append(ticker)
            continue
        price = float(close.iloc[-1])
        invested = qty * buy
        value = qty * price
        rows.append({
            "symbol": ticker,
            "sector": LARGE_CAPS.get(sym.replace(".NS", ""), "Other"),
            "quantity": qty,
            "buy_price": safe_round(buy),
            "current_price": safe_round(price),
            "invested": safe_round(invested),
            "current_value": safe_round(value),
            "profit_loss": safe_round(value - invested),
            "profit_loss_pct": safe_round((value / invested - 1) * 100),
        })

    if not rows:
        raise HTTPException(status_code=404, detail=f"No price data found for: {not_found}")

    total_invested = sum(r["invested"] for r in rows)
    total_value = sum(r["current_value"] for r in rows)
    for r in rows:
        r["weight_pct"] = safe_round(r["current_value"] / total_value * 100)

    sector_split = {}
    for r in rows:
        sector_split[r["sector"]] = sector_split.get(r["sector"], 0) + r["weight_pct"]

    biggest = max(rows, key=lambda r: r["weight_pct"])
    warning = None
    if biggest["weight_pct"] > 40:
        warning = f"{biggest['symbol']} is {biggest['weight_pct']}% of your portfolio. Too much money in one stock raises risk."

    return {
        "holdings": rows,
        "not_found": not_found,
        "total_invested": safe_round(total_invested),
        "total_current_value": safe_round(total_value),
        "total_profit_loss": safe_round(total_value - total_invested),
        "total_profit_loss_pct": safe_round((total_value / total_invested - 1) * 100),
        "best_holding": max(rows, key=lambda r: r["profit_loss_pct"])["symbol"],
        "worst_holding": min(rows, key=lambda r: r["profit_loss_pct"])["symbol"],
        "sector_split_pct": {k: safe_round(v) for k, v in sector_split.items()},
        "concentration_warning": warning,
    }
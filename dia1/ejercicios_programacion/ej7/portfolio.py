import pandas as pd
import numpy as np


class Portfolio:

    def __init__(self):
        self._h = {}
        self._p = {}

    def add_stock(self, ticker, qty, px):
        if ticker in self._h:
            old_q = self._h[ticker]
            old_p = self._p[ticker]
            self._p[ticker] = (old_p * old_q + px * qty) / (old_q + qty)
            self._h[ticker] = old_q + qty
        else:
            self._h[ticker] = qty
            self._p[ticker] = px

    def remove_stock(self, ticker, qty=None):
        if ticker not in self._h:
            raise ValueError(f"Ticker {ticker} not in portfolio")
        if qty is None or qty >= self._h[ticker]:
            del self._h[ticker]
            del self._p[ticker]
        else:
            self._h[ticker] -= qty

    def total_value(self):
        val = 0
        for t in self._h:
            val += self._h[t] * self._p[t]
        return val

    def weights(self):
        tv = self.total_value()
        if tv == 0:
            return {}
        w = {}
        for t in self._h:
            w[t] = (self._h[t] * self._p[t]) / tv
        return w

    def holdings(self):
        result = []
        for t in self._h:
            result.append({
                'ticker': t,
                'qty': self._h[t],
                'avg_price': self._p[t],
                'mkt_value': self._h[t] * self._p[t]
            })
        return pd.DataFrame(result)

    def update_prices(self, prices_dict):
        for t, px in prices_dict.items():
            if t in self._p:
                self._p[t] = px

    def daily_returns(self, price_history):
        tickers = [t for t in self._h if t in price_history.columns]
        if not tickers:
            return pd.Series(dtype=float)

        w = self.weights()
        port_ret = pd.Series(0.0, index=price_history.index[1:])

        for t in tickers:
            px = price_history[t]
            r = px.pct_change().dropna()
            port_ret = port_ret + w[t] * r

        return port_ret

    def volatility(self, price_history, window=30):
        ret = self.daily_returns(price_history)
        if len(ret) < window:
            return None
        return ret.rolling(window).std().dropna()

    def cumulative_return(self, price_history):
        ret = self.daily_returns(price_history)
        return (1 + ret).cumprod() - 1

    def max_drawdown(self, price_history):
        cum = self.cumulative_return(price_history) + 1
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()

    def __repr__(self):
        n = len(self._h)
        tv = self.total_value()
        return f"Portfolio({n} stocks, value={tv:,.2f})"

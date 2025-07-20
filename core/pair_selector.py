# core/pair_selector.py

from collections import defaultdict
import math, random
from io.binance_client import client  # kita bisa pakai ccxt/python-binance

class DynamicUCBSelector:
    def __init__(self, k: int, static_pairs: list[str], universe_size: int):
        self.k = k
        self.static = static_pairs
        self.universe_size = universe_size
        self.N_total = 0
        self.stats = defaultdict(lambda: {'N':0, 'reward':0.0})

    def update_reward(self, pair: str, sharpe: float):
        st = self.stats[pair]
        st['N'] += 1; st['reward'] += sharpe
        self.N_total += 1

    def _fetch_top_universe(self) -> list[str]:
        """
        Ambil top-N pair berdasar 24h volume dari Binance.
        """
        # via CCXT: markets = exchange.fetch_tickers()
        tickers = client.get_ticker_24hr()  # list of dicts
        # filter hanya USDT pairs dan sort by quoteVolume
        usdt = [t for t in tickers if t['symbol'].endswith("USDT")]
        sorted_ = sorted(usdt, key=lambda x: float(x['quoteVolume']), reverse=True)
        return [t['symbol'] for t in sorted_[:self.universe_size]]

    def choose(self) -> list[str]:
        # 1. bangun universe dinamis
        universe = self._fetch_top_universe()

        # 2. hitung skor UCB untuk tiap pair di universe
        scores = {}
        for p in universe:
            st = self.stats[p]
            if st['N'] == 0:
                scores[p] = float('inf')
            else:
                mean_r = st['reward']/st['N']
                bonus = math.sqrt(2 * math.log(self.N_total) / st['N'])
                scores[p] = mean_r + bonus

        # 3. pilih top-k
        topk = sorted(scores, key=scores.get, reverse=True)[:self.k]
        if not topk:
            topk = random.sample(universe, self.k)

        # 4. gabung static + dynamic
        return list(self.static) + topk

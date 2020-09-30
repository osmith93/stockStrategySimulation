import random
import matplotlib.pyplot as plt
import numpy as np
import csv

class Strategy:
    def __init__(self, savings_per_block=100, pay_period=10):
        self.shares = [0]
        self.cash = [0]
        self.last_buying_day = 0
        self.savings_per_block = savings_per_block
        self.pay_period = pay_period
        self.today = 0

    def buy_shares(self, current_stock_price):
        self.last_buying_day = self.today
        current_cash = self.cash[-1]
        current_shares = self.shares[-1]
        new_shares = current_cash / current_stock_price
        self.shares.append(current_shares + new_shares)
        self.cash[-1] = 0

    def dont_buy_shares(self):
        current_shares = self.shares[-1]
        self.shares.append(current_shares)

    def get_income(self):
        self.cash.append(self.cash[-1])
        if self.today % self.pay_period == 0:
            self.cash[-1] += self.savings_per_block

    def i_will_buy_shares(self, stock_history):
        pass

    def next_day(self, stock_history):
        self.get_income()
        if (self.i_will_buy_shares(stock_history)):
            self.buy_shares(stock_history[-1])
        else:
            self.dont_buy_shares()

        self.today += 1

    def value_history(self, stock_history):
        return [current_cash + current_shares * current_stock_price for
                current_cash, current_shares, current_stock_price in
                zip(self.cash[1:], self.shares[1:], stock_history)]


class BenchmarkStrategy(Strategy):
    def i_will_buy_shares(self, stock_history):
        return True


class ModifiedStrategy(Strategy):
    def __init__(self, threshhold=0.9):
        super().__init__()
        self.threshhold = threshhold

    def i_will_buy_shares(self, stock_history):
        if (self.today > 0):
            current_stock_price = stock_history[-1]
            last_maximum_stock_price = np.max(stock_history[self.last_buying_day:self.today])
            if current_stock_price < last_maximum_stock_price * self.threshhold:
                return True
        return False


def generate_stock_history(n=10000):
    def next_step(now, tendency=1e-5, volatility=100):
        return now * (1 + tendency * random.normalvariate(1, volatility))

    timestamps = range(n)
    stock_history = [100.0]
    for t in timestamps:
        stock_history.append(next_step(stock_history[t]))

    return timestamps, stock_history


def read_stock_history(filename="GDAXI.csv", column_name="Open"):
    stock_history = []
    with open(filename) as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                column_id = row.index(column_name)
            else:
                if row[column_id] != 'null':
                    stock_history.append(float(row[column_id]))
                else:
                    stock_history.append(stock_history[-1])
    timestamps = range(len(stock_history))
    return timestamps, stock_history


def applyStrategy(strategy, stock_history):
    timestamps = range(len(stock_history))
    for t in timestamps:
        strategy.next_day(stock_history[:t + 1])


timestamps, stock_history = generate_stock_history()
#timestamps, stock_history = read_stock_history()


noStrategy = Strategy()
benchmarkStrategy = BenchmarkStrategy()
modifiedStrategy = ModifiedStrategy(threshhold=0.98)

applyStrategy(benchmarkStrategy, stock_history)
applyStrategy(modifiedStrategy, stock_history)
applyStrategy(noStrategy, stock_history)

print(f"The benchmark strategy yielded {benchmarkStrategy.value_history(stock_history)[-1]}.")
print(f"The modified strategy yielded {modifiedStrategy.value_history(stock_history)[-1]} ({modifiedStrategy.cash[-1]} in cash).")

fig, ax = plt.subplots(1, 3)
ax[0].plot(stock_history)
ax[0].set_title("Stock price")
ax[1].plot(benchmarkStrategy.value_history(stock_history))
ax[1].set_title("Benchmark strategy")
ax[2].plot(modifiedStrategy.value_history(stock_history))
ax[2].plot(modifiedStrategy.cash)
ax[2].set_title("Modified strategy")

fig.show()

# import numpy as np

# # Overclocked Strategy - Interim Version
# # This strategy uses PCA to remove the shared market effect, calculates momentum from residuals,
# # applies Z-score filtering, and ranks stocks based on momentum to create long/short positions.
# # This code is designed to be run in a trading environment where `prcSoFar` is provided as input.
# # prcSoFar is a 2D numpy array where rows are instruments and columns are time points (prices)

# # number of instruments (stocks)
# nInst = 50

# # global variables to track current positions and how long we’ve held them
# currentPos = np.zeros(nInst)
# hold_days = np.zeros(nInst)

# def getMyPosition(prcSoFar):
#     global currentPos, hold_days
#     n, t = prcSoFar.shape

#     # if we don’t have enough data (less than 1 year), don’t trade
#     if t < 250:
#         currentPos = np.zeros(n)
#         hold_days = np.zeros(n)
#         return currentPos

#     # === 1. check if the market is in an uptrend ===
#     # we calculate the average price of all instruments
#     market_prices = prcSoFar.mean(axis=0)
#     # check the 21-day market return (approx. 1 trading month)
#     market_return = np.log(market_prices[-1] / market_prices[-21])

#     # if the market is falling, we sit out (no trades)
#     if market_return <= 0 or np.std(market_prices[-21:]) > np.std(market_prices[-252:]):
#         currentPos = np.zeros(n)
#         hold_days = np.zeros(n)
#         return currentPos

#     # === 2. Calculate daily log returns ===
#     returns = np.diff(np.log(prcSoFar), axis=1)

#     # === 3. use PCA to remove the shared market effect ===
#     X = returns[:, -252:]  # Use past 252 days (1 year)
#     X -= X.mean(axis=1, keepdims=True)  # Demean each stock's return series
#     cov = np.cov(X)  # Covariance matrix
#     _, eigvecs = np.linalg.eigh(cov)  # Eigen decomposition(i, ignore the first value)
#     market_pc = eigvecs[:, -1].reshape(-1, 1)  # First principal component (market direction)
#     market_component = market_pc @ (market_pc.T @ X)  # Project market factor out
#     residuals = X - market_component  # Leftover = unique stock behavior

#     # === 4. Calculate momentum from residuals ===
#     # Momentum = average residual return over past 20 days
#     momentum = residuals[:, -20:].mean(axis=1)

#     # === 5. Apply Z-score filtering to only keep strong momentum stocks ===
#     z = (momentum - momentum.mean()) / (momentum.std() + 1e-6)

#     # Changed to 1 to filter out weak signals
#     momentum[z <= 1.1] = 0  # Filter out weak signals

#     # === 6. Calculate 60-day EWMA volatility ===
#     lam = 0.94  # Smoothing factor
#     ewma_weights = np.power(lam, np.arange(60)[::-1])  # Higher weight to recent days
#     ewma_weights /= ewma_weights.sum()
#     ewma_vol = np.sqrt(np.sum((residuals[:, -60:] ** 2) * ewma_weights, axis=1) + 1e-8)

#     # === 7. Filter out the most volatile stocks (top 15%) ===
#     vol_cutoff = np.percentile(ewma_vol, 85)
#     momentum[ewma_vol >= vol_cutoff] = 0  # Don’t trade these
#     ewma_vol[ewma_vol >= vol_cutoff] = 1e6  # Prevent them from getting capital

#     # === 8. Rank the momentum signals and choose top/bottom 5 ===
#     ranked = np.argsort(momentum)
#     longs = [i for i in ranked[::-1] if momentum[i] > 0][:5]  # Top 5
#     shorts = [i for i in ranked if momentum[i] < 0][:5]       # Bottom 5

#     # === 9. Allocate positions ===
#     pos = np.zeros(n)
#     capital = 1_000_000  # Total capital
#     max_dollars = 10_000  # Max per stock

#     for i in range(n):
#         # === Keep current position for 1 more day if it's not in new long/short ===
#         if currentPos[i] != 0 and i not in longs + shorts:
#             if hold_days[i] < 2:  # If held for less than 2 days, keep it
#                 pos[i] = currentPos[i]
#                 hold_days[i] += 1
#             else:
#                 hold_days[i] = 0  # Exit
#         # === Allocate new long positions ===
#         elif i in longs:
#             total_inv = sum(1 / ewma_vol[j] for j in longs)
#             weight = capital / (2 * total_inv)
#             dollars = min(weight / ewma_vol[i], max_dollars)
#             pos[i] = dollars / prcSoFar[i, -1]
#             hold_days[i] = 0
#         # === Allocate new short positions ===
#         elif i in shorts:
#             total_inv = sum(1 / ewma_vol[j] for j in shorts)
#             weight = capital / (2 * total_inv)
#             dollars = min(weight / ewma_vol[i], max_dollars)
#             pos[i] = -dollars / prcSoFar[i, -1]
#             hold_days[i] = 0
#         else:
#             hold_days[i] = 0  # If doing nothing, reset hold count

#     # === 10. Clip and return final positions ===
#     currentPos = np.clip(np.round(pos), -1000, 1000)
    
#     return currentPos

# =====
# mean(PL): -5.5
# return: -0.00074
# StdDev(PL): 134.95
# annSharpe(PL): -0.64
# totDvolume: 1494242
# Score: -18.96

import numpy as np

nInst = 50
currentPos = np.zeros(nInst)
hold_days = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos, hold_days
    n, t = prcSoFar.shape
    if t < 250:
        currentPos = np.zeros(nInst)
        hold_days = np.zeros(nInst)
        return currentPos

    returns = np.diff(np.log(prcSoFar), axis=1)
    # Smooth burst with 3-day rolling average
    burst_raw = np.log(prcSoFar[:, -1] / prcSoFar[:, -6])
    burst = (burst_raw + np.roll(burst_raw,1) + np.roll(burst_raw,2))/3

    trend = np.log(prcSoFar[:, -1] / prcSoFar[:, -60])
    vol = np.std(returns[:, -60:], axis=1)
    vol_cut = np.percentile(vol, 80)

    signal_thresh = np.percentile(np.abs(burst), 80)

    valid_long = (burst > signal_thresh) & (trend > 0.05) & (vol < vol_cut)
    valid_short = (burst < -signal_thresh) & (trend < -0.05) & (vol < vol_cut)

    longs = [i for i in np.argsort(burst)[::-1] if valid_long[i]][:1]
    shorts = [i for i in np.argsort(burst) if valid_short[i]][:0]  # only longs to reduce turnover

    capital = 1_000_000
    max_dollars = 10_000

    pos = np.zeros(nInst)
    if longs:
        i = longs[0]
        pos[i] = min(max_dollars, capital * 0.5) / prcSoFar[i, -1]
        hold_days[i] = 1

    # Stop loss 1.5% + hold winners max 7 days
    for i in range(nInst):
        if currentPos[i] != 0:
            entry_price = prcSoFar[i, -2] if t >= 2 else prcSoFar[i, -1]
            cur_price = prcSoFar[i, -1]
            if currentPos[i] > 0 and (cur_price < 0.985 * entry_price):
                pos[i] = 0
                hold_days[i] = 0

    for i in range(nInst):
        if pos[i] == 0 and currentPos[i] != 0:
            if hold_days[i] < 5:
                pos[i] = currentPos[i]
                hold_days[i] += 1
            else:
                hold_days[i] = 0
        elif pos[i] == 0:
            hold_days[i] = 0

    currentPos = np.clip(np.round(pos), -1000, 1000)
    return currentPos

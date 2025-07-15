# ------ General Round Submission Code ------
import numpy as np

nInst = 50
currentPos = np.zeros(nInst)
hold_days = np.zeros(nInst)
entry_prices = np.zeros(nInst)

def getMyPosition(prcSoFar):
    # === PARAMETERS ===
    minN = 3
    maxN = 5
    burst_percentile = 81.6
    trend_thresh = 0.00025
    vol_percentile = 83
    base_target_risk = 0.01
    max_dollars = 10000
    max_port_vol = 0.0013  # Stricter portfolio volatility cap
    corr_thresh = 0.8
    stop_loss_mult = 1.1  # Tighter stop-loss
    take_profit_mult = 3.0  # Let winners run a bit more
    burst_weight = 0.7
    momentum_weight = 0.29

    global currentPos, hold_days, entry_prices
    n, t = prcSoFar.shape
    if t < 250:
        currentPos = np.zeros(nInst)
        hold_days = np.zeros(nInst)
        entry_prices = np.zeros(nInst)
        return currentPos

    returns = np.diff(np.log(prcSoFar), axis=1)
    burst_raw = np.log(prcSoFar[:, -1] / prcSoFar[:, -6])
    burst = (burst_raw + np.roll(burst_raw, 1) + np.roll(burst_raw, 2)) / 3
    trend = np.log(prcSoFar[:, -1] / prcSoFar[:, -60])
    vol = np.std(returns[:, -38:], axis=1)
    vol_cut = np.percentile(vol, vol_percentile)
    signal_thresh = np.percentile(np.abs(burst), burst_percentile)

    # === Momentum Signal (20-day return) ===
    momentum = np.zeros(nInst)
    if t >= 21:
        momentum = np.log(prcSoFar[:, -1] / prcSoFar[:, -21])
    momentum = np.clip(momentum / (np.max(np.abs(momentum)) + 1e-8), -1, 1)

    # === Ensemble Signal ===
    ensemble_signal = burst_weight * burst + momentum_weight * momentum

    valid_long = (ensemble_signal > np.percentile(ensemble_signal, burst_percentile)) & (trend > trend_thresh) & (vol < vol_cut)
    valid_short = (ensemble_signal < -np.percentile(np.abs(ensemble_signal), burst_percentile)) & (trend < -trend_thresh) & (vol < vol_cut)

    N_long = min(maxN, max(minN, np.sum(valid_long)))
    N_short = min(maxN, max(minN, np.sum(valid_short)))

    lookback = min(60, t-1)
    corr_matrix = np.corrcoef(returns[:, -lookback:]) if lookback > 1 else np.eye(nInst)

    def select_diversified(indices, N):
        selected = []
        for i in indices:
            if any(abs(corr_matrix[i, j]) > corr_thresh for j in selected):
                continue
            selected.append(i)
            if len(selected) >= N:
                break
        return selected

    # Prioritize high-conviction, uncorrelated trades
    long_candidates = [i for i in np.argsort(-ensemble_signal) if valid_long[i]]
    short_candidates = [i for i in np.argsort(ensemble_signal) if valid_short[i]]

    longs = select_diversified(long_candidates, N_long)
    shorts = select_diversified(short_candidates, N_short)

    capital = 1_000_000
    pos = np.zeros(nInst)

    # Non-linear scaling for strongest signals
    for i in longs:
        dollar_vol = vol[i] * prcSoFar[i, -1]
        # Use squared signal strength for more aggressive sizing of best signals
        signal_strength = ((ensemble_signal[i] - np.percentile(ensemble_signal, burst_percentile)) / (np.max(np.abs(ensemble_signal)) + 1e-8)) ** 2
        size = min(max_dollars, capital * base_target_risk * (1 + signal_strength) / (dollar_vol + 1e-8))
        pos[i] = size / prcSoFar[i, -1]
        hold_days[i] = 1
        entry_prices[i] = prcSoFar[i, -1]
    for i in shorts:
        dollar_vol = vol[i] * prcSoFar[i, -1]
        signal_strength = ((abs(ensemble_signal[i]) - np.percentile(np.abs(ensemble_signal), burst_percentile)) / (np.max(np.abs(ensemble_signal)) + 1e-8)) ** 2
        size = min(max_dollars, capital * base_target_risk * (1 + signal_strength) / (dollar_vol + 1e-8))
        pos[i] = -size / prcSoFar[i, -1]
        hold_days[i] = 1
        entry_prices[i] = prcSoFar[i, -1]

    for i in range(nInst):
        if currentPos[i] != 0:
            entry_price = entry_prices[i] if entry_prices[i] > 0 else prcSoFar[i, -1]
            cur_price = prcSoFar[i, -1]
            v = vol[i]
            stop_loss = stop_loss_mult * v * entry_price
            take_profit = take_profit_mult * v * entry_price
            if (currentPos[i] > 0 and cur_price < entry_price - stop_loss) or \
               (currentPos[i] < 0 and cur_price > entry_price + stop_loss):
                pos[i] = 0
                hold_days[i] = 0
                entry_prices[i] = 0
            elif (currentPos[i] > 0 and cur_price > entry_price + take_profit) or \
                 (currentPos[i] < 0 and cur_price < entry_price - take_profit):
                pos[i] = 0
                hold_days[i] = 0
                entry_prices[i] = 0

        if pos[i] == 0 and currentPos[i] != 0:
            if hold_days[i] < 6:
                pos[i] = currentPos[i]
                hold_days[i] += 1
            else:
                hold_days[i] = 0
                entry_prices[i] = 0
        elif pos[i] == 0:
            hold_days[i] = 0
            entry_prices[i] = 0

    weights = pos * prcSoFar[:, -1] / capital
    port_vol = np.sqrt(np.sum((weights * vol) ** 2))
    if port_vol > max_port_vol:
        pos *= max_port_vol / (port_vol + 1e-8)

    currentPos = np.clip(np.round(pos), -1000, 1000)
    return currentPos

# =====
# mean(PL): 117.6
# return: 0.00624
# StdDev(PL): 409.18
# annSharpe(PL): 4.53
# totDvolume: 3742542
# Score: 76.66
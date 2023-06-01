def extract_flex_metrics(baseline_power, flexible_power, timeperiod):
    charging = np.sum(np.where(flexible_power > baseline_power, flexible_power - baseline_power, 0))
    discharging = np.sum(np.where(flexible_power < baseline_power, baseline_power - flexible_power, 0))

    rte = discharging / charging
    ed = discharging / np.sum(baseline_power)
    pd = discharging / (timeperiod * np.mean(baseline_power))
    return rte, ed, pd 
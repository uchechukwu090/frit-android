export class DataBudgetManager {
  constructor(config = {}) {
    this.limits = { fcs_daily: config.fcs_daily || 80, fcs_minute: config.fcs_minute || 8, td_daily: config.td_daily || 400 };
    this.counters = { fcs_daily: 0, fcs_minute: 0, td_daily: 0, lastMinuteReset: Date.now(), lastDailyReset: Date.now() };
    this.rotationIndex = 0;
    this.symbolPool = config.symbolPool || ["EURUSD","GBPUSD","USDJPY","AUDUSD","XAUUSD","BTCUSD","ETHUSD","SOLUSD"];
  }
  _resetIfNeeded() {
    const now = Date.now();
    if (now - this.counters.lastDailyReset > 24 * 60 * 60 * 1000) { this.counters.fcs_daily = 0; this.counters.td_daily = 0; this.counters.lastDailyReset = now; }
    if (now - this.counters.lastMinuteReset > 60 * 1000) { this.counters.fcs_minute = 0; this.counters.lastMinuteReset = now; }
  }
  canCall(api) {
    this._resetIfNeeded();
    if (api === "fcs") return this.counters.fcs_daily < this.limits.fcs_daily && this.counters.fcs_minute < this.limits.fcs_minute;
    if (api === "twelve") return this.counters.td_daily < this.limits.td_daily;
    return true;
  }
  recordCall(api) {
    this._resetIfNeeded();
    if (api === "fcs") { this.counters.fcs_daily++; this.counters.fcs_minute++; }
    if (api === "twelve") { this.counters.td_daily++; }
  }
  getRemaining(api) {
    this._resetIfNeeded();
    if (api === "fcs") return { daily: this.limits.fcs_daily - this.counters.fcs_daily, minute: this.limits.fcs_minute - this.counters.fcs_minute };
    if (api === "twelve") return { daily: this.limits.td_daily - this.counters.td_daily };
    return { daily: 0 };
  }
  getRotatedBatch(batchSize = 3) {
    const batch = [];
    for (let i = 0; i < batchSize; i++) { batch.push(this.symbolPool[this.rotationIndex % this.symbolPool.length]); this.rotationIndex++; }
    return batch;
  }
  getStatus() {
    return { fcs_remaining: this.getRemaining("fcs"), td_remaining: this.getRemaining("twelve"), rotation_index: this.rotationIndex, pool_size: this.symbolPool.length };
  }
}

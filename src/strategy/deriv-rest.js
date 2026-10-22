// ============================================================================
// DERIV REST — real endpoints of the Deriv "Options" API (api.derivws.com)
// ----------------------------------------------------------------------------
// MIGRATED (Oct 2026): the legacy REST surface (account settings, statistics,
// self-exclusion, cashier, app registration) does NOT exist on the current
// API — verified live: /account/v1/settings|statistics|self-exclusion,
// /cashier/v1/info, /applications/v1/list|get|register|update|delete and
// /account/v1/statement all return 404. This module only exposes what actually
// exists (route probed live, HTTP 401/422 = exists):
//   GET  /v1/health                                            (public)
//   GET  /trading/v1/options/accounts                          (Bearer)
//   POST /trading/v1/options/accounts/{accountId}/otp          (Bearer)
//   GET  /trading/v1/options/legacy/statement                  (Bearer)
//   GET  /wallet/v1/wallets                                    (old /accounts 404s)
//   GET  /payment-agents/v1/agents?country=...
//   POST /payment-agents/v1/withdraw                            (verification code)
//   GET  /applications/v1/markup-statistics?date_from&date_to
//   POST /account/v1/nickname
//
// SAFETY:
//   - Read-first; the one money-moving method (payment-agent withdrawal) is
//     named explicitly and the API itself enforces the SMS verification code.
//   - PAT auth needs the Deriv-App-ID header — sent automatically.
// ============================================================================

const REST_BASE = "https://api.derivws.com";

export class DerivRestClient {
  constructor({
    token = process.env.DERIV_TOKEN
      || process.env.DERIV_DEMO_TOKEN
      || process.env.DERIV_REAL_TOKEN
      || "",
    appId = process.env.DERIV_APP_ID || "",
    fetchImpl = fetch,
  } = {}) {
    this.token = token;
    this.appId = String(appId || "");
    this.fetch = fetchImpl;
  }

  _headers() {
    const H = { "Content-Type": "application/json" };
    if (this.token) {
      H.Authorization = `Bearer ${this.token}`;
      H["Deriv-App-ID"] = this.appId;
    }
    return H;
  }

  async rest(method, path, body = null, { timeoutMs = 20_000 } = {}) {
    if (!this.token) return { error: { message: "no DERIV_TOKEN — set it in server/.env" } };
    try {
      const res = await this.fetch(`${REST_BASE}${path}`, {
        method,
        headers: this._headers(),
        body: body != null ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(timeoutMs),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const err = data?.errors?.[0] || data?.error || data?.detail || {};
        return { error: { message: err.message || err.detail?.message || `HTTP ${res.status}`, code: err.code } };
      }
      return data;
    } catch (e) {
      return { error: { message: e.message } };
    }
  }

  /** Public service health check (no auth). */
  health() {
    return this._raw("GET", "/v1/health", null, true);
  }

  /** Unauthenticated-capable variant (token optional). */
  async _raw(method, path, body = null, allowNoToken = false, timeoutMs = 20_000) {
    if (!this.token && !allowNoToken) return { error: { message: "no DERIV_TOKEN — set it in server/.env" } };
    try {
      const res = await this.fetch(`${REST_BASE}${path}`, {
        method,
        headers: this._headers(),
        body: body != null ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(timeoutMs),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const err = data?.errors?.[0] || data?.error || data?.detail || {};
        return { error: { message: err.message || err.detail?.message || `HTTP ${res.status}`, code: err.code } };
      }
      return data;
    } catch (e) {
      return { error: { message: e.message } };
    }
  }

  // ---- trading accounts (drives the websocket OTP flow in DerivClient) ----
  async accounts() { return this.rest("GET", "/trading/v1/options/accounts"); }
  async otpUrl(accountId) { return this.rest("POST", `/trading/v1/options/accounts/${encodeURIComponent(accountId)}/otp`); }

  /** Trade history for the account (legacy-named route on the new platform). */
  async statement({ limit = 50, offset = 0 } = {}) {
    return this.rest("GET", `/trading/v1/options/legacy/statement?limit=${limit}&offset=${offset}`);
  }

  // ---- wallet ----
  async wallets() { return this.rest("GET", "/wallet/v1/wallets"); }

  // ---- payments ----
  async paymentAgents(countryCode) {
    return this.rest("GET", `/payment-agents/v1/agents?country=${encodeURIComponent(countryCode)}`);
  }

  /**
   * Withdrawal through a payment agent. The API requires a verification code
   * sent to the client's registered contact; we never attempt to skip that.
   */
  async requestWithdrawal({ amount, currency, agentId, verificationCode, dryRun = false }) {
    return this.rest("POST", "/payment-agents/v1/withdraw", {
      amount, currency, agent_id: agentId, verification_code: verificationCode,
      ...(dryRun ? { dry_run: 1 } : {}),
    });
  }

  // ---- application management ----
  async appMarkupStatistics(dateFrom, dateTo) {
    return this.rest("GET", `/applications/v1/markup-statistics?date_from=${dateFrom}&date_to=${dateTo}`);
  }

  // ---- misc ----
  async updateNickname(nickname) { return this.rest("POST", "/account/v1/nickname", { nickname }); }
}

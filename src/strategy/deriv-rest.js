// ============================================================================
// DERIV REST — account management, payments, application insights
// ----------------------------------------------------------------------------
// DerivClient only speaks WebSocket (data + trade). Account settings, payment
// agents, wallets, and app markup statistics are REST on
// https://api.derivws.com with an OAuth2 Bearer token (or PAT + Deriv-App-ID
// header). This module keeps DerivClient as-is and gives it a `rest()` for
// everything else.
//
// SAFETY:
//   - All endpoints are read-first; mutating ones (self-exclusion, account
//     closure, withdrawal) live behind explicit named methods so a stray call
//     cannot move money or close an account.
//   - Withdrawal requires a verification code (phone/SMS) — the API itself
//     enforces that; we never attempt to bypass it.
//   - App registration requires the `trade` scope; markup statistics require
//     application_read + Deriv-App-ID. Everything is opted-in via env.
// ============================================================================

const REST_BASE = "https://api.derivws.com";

export class DerivRestClient {
  constructor({
    token = process.env.DERIV_TOKEN
      || process.env.DERIV_DEMO_TOKEN
      || process.env.DERIV_REAL_TOKEN
      || "",
    appId = process.env.DERIV_APP_ID || "1089",
    fetchImpl = fetch,
  } = {}) {
    this.token = token;
    this.appId = String(appId || "1089");
    this.fetch = fetchImpl;
  }

  _headers() {
    const H = { "Content-Type": "application/json" };
    if (this.token) H.Authorization = `Bearer ${this.token}`;
    if (this.token) H["Deriv-App-ID"] = this.appId;
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
        const err = data?.errors?.[0] || data?.error || { message: `HTTP ${res.status}` };
        return { error: { message: err.message || err.detail?.message || `HTTP ${res.status}`, code: err.code } };
      }
      return data;
    } catch (e) {
      return { error: { message: e.message } };
    }
  }

  // ---- account management (read) ----
  async getSettings() { return this.rest("GET", "/account/v1/settings"); }
  async getAccountStatistics() { return this.rest("GET", "/account/v1/statistics"); }
  async getFinancialAssessment() { return this.rest("GET", "/account/v1/financial-assessment"); }
  async getResidenceList() { return this.rest("GET", "/residence/v1/list"); }
  async getLandingCompany() { return this.rest("GET", "/landingcompany/v1/details"); }
  async getRealityCheck() { return this.rest("GET", "/reality-check/v1/summary"); }

  // ---- account management (mutating — named so they cannot be called by accident) ----
  async updateSettings(patch) { return this.rest("PATCH", "/account/v1/settings", patch); }
  async submitFinancialAssessment(answers) { return this.rest("POST", "/account/v1/financial-assessment", answers); }
  async setSelfExclusion(params) { return this.rest("POST", "/account/v1/self-exclusion", params); }
  async getSelfExclusion() { return this.rest("GET", "/account/v1/self-exclusion"); }
  async requestAccountClosure(reason, verificationCode) {
    return this.rest("POST", "/account/v1/close", { reason, verification_code: verificationCode });
  }
  async requestAccountReopen(reason, verificationCode) {
    return this.rest("POST", "/account/v1/reopen", { reason, verification_code: verificationCode });
  }
  async logout() { return this.rest("POST", "/account/v1/logout", {}); }

  // ---- payments ----
  async paymentAgents(countryCode) { return this.rest("GET", `/payment-agents/v1/agents?country=${encodeURIComponent(countryCode)}`); }
  async cashierInfo(currency) { return this.rest("GET", `/cashier/v1/info?currency=${encodeURIComponent(currency)}`); }
  async wallets() { return this.rest("GET", "/wallet/v1/accounts"); }

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

  async statement({ limit = 50, offset = 0 } = {}) {
    return this.rest("GET", `/account/v1/statement?limit=${limit}&offset=${offset}`);
  }

  // ---- application management ----
  async appList() { return this.rest("GET", "/applications/v1/list"); }
  async appGet(appId) { return this.rest("GET", `/applications/v1/get?app_id=${appId}`); }
  async appMarkupStatistics(dateFrom, dateTo) {
    return this.rest("GET", `/applications/v1/markup-statistics?date_from=${dateFrom}&date_to=${dateTo}`);
  }

  /**
   * Register an OAuth application. Requires the trade scope and a redirect URI;
   * the returned app_id/secret are what mark THIS app as a Deriv-connected app
   * (markup commission, not just API access).
   */
  async appRegister({ name, redirectUri, scopes, appMarkUp = 0, verificationCode }) {
    return this.rest("POST", "/applications/v1/register", {
      name, redirect_uri: redirectUri, scopes, app_mark_up: appMarkUp, verification_code: verificationCode,
    });
  }

  async appUpdate(appId, patch) { return this.rest("PATCH", `/applications/v1/update?app_id=${appId}`, patch); }
  async appDelete(appId) { return this.rest("POST", `/applications/v1/delete?app_id=${appId}`); }
}

// ============================================================================
// FRIT BILLING — credit ledger, entitlement checks, provider-job billing
// ----------------------------------------------------------------------------
// Extracted from index.js (audit Week 4, "entitlement/credit middleware
// extraction") so the money logic is unit-testable with an in-memory SQLite db.
// Behavior is byte-for-byte the same as the inline versions it replaced:
//   - checkEntitlement never debits; it only gates (plan feature + balance).
//   - debitCredits is a single guarded UPDATE (no double-spend), post-success.
//   - chargeProviderJob debits at most once (provider_jobs.charged flag).
// index.js keeps thin wrapper functions with the same names, so no route code
// changed. Dependencies are injected so tests need none of the server.
// ============================================================================

/**
 * @param {object} deps
 * @param {object} deps.db                 better-sqlite3 database
 * @param {object} deps.SUB_TIERS          { tier: { credits_mo, features[] } }
 * @param {(user_id:string)=>string} deps.activeTier
 * @param {()=>string} deps.monthStr       e.g. "2026-10"
 * @param {object} deps.CREDIT_COSTS       charge code -> credits
 * @param {object} deps.FEATURE_CHARGE     feature -> charge code
 * @param {object} deps.FEATURE_LABELS     feature -> human label
 */
export function createBilling(deps) {
  const { db, SUB_TIERS, activeTier, monthStr, CREDIT_COSTS, FEATURE_CHARGE, FEATURE_LABELS } = deps;

  function creditBalance(user_id) {
    const row = db.prepare("SELECT balance FROM user_credits WHERE user_id = ?").get(user_id);
    return row?.balance ?? 0;
  }

  function creditTxn(user_id, amount, kind, note = "") {
    const run = db.transaction(() => {
      db.prepare("INSERT INTO credit_txns (user_id, amount, kind, note) VALUES (?, ?, ?, ?)").run(user_id, amount, kind, note);
      db.prepare("INSERT INTO user_credits (user_id, balance, total_earned, total_spent, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(user_id) DO UPDATE SET balance = balance + ?, total_earned = total_earned + MAX(0, ?), total_spent = total_spent + MAX(0, -?), updated_at = CURRENT_TIMESTAMP").run(user_id, amount, Math.max(0, amount), Math.max(0, -amount), amount, amount, amount);
    });
    run();
    return creditBalance(user_id);
  }

  function grantMonthlyCredits(user_id) {
    const tier = activeTier(user_id);
    const t = SUB_TIERS[tier];
    if (!t?.credits_mo) return creditBalance(user_id);
    const month = monthStr();
    const grantKey = `${tier}:${month}`; // tier-scoped: a mid-month upgrade still receives the new plan's credits
    const row = db.prepare("SELECT plan_grant_month FROM user_credits WHERE user_id = ?").get(user_id);
    if (row?.plan_grant_month === grantKey) return creditBalance(user_id);
    // Anti-farming: monthly credits only flow to REAL installs (a key issued by
    // /auth/install for this install_id) or paying subscribers.
    const hasKey = db.prepare("SELECT 1 FROM api_keys WHERE user_id = ?").get(user_id);
    const hasSub = db.prepare("SELECT 1 FROM subscriptions WHERE user_id = ?").get(user_id);
    if (!hasKey && !hasSub) return creditBalance(user_id);
    creditTxn(user_id, t.credits_mo, "plan_grant", `${tier} ${month}`);
    db.prepare("UPDATE user_credits SET plan_grant_month = ? WHERE user_id = ?").run(grantKey, user_id);
    return creditBalance(user_id);
  }

  function planAllows(tier, feature) {
    const f = SUB_TIERS[tier]?.features || [];
    if (feature === "stt") return f.includes("chat"); // voice INPUT is part of chat; "voice" feature = creator voice tools
    return f.includes(feature);
  }

  function checkEntitlement(user_id, feature) {
    grantMonthlyCredits(user_id);
    const tier = activeTier(user_id);
    if (!planAllows(tier, feature)) {
      const need = tier === "free" ? `a paid plan with ${FEATURE_LABELS[feature] || feature}` : `a plan that includes ${FEATURE_LABELS[feature] || feature}`;
      return { ok: false, code: "feature_locked", message: `${FEATURE_LABELS[feature] || feature} isn't in your ${tier} plan — upgrade to ${need}.` };
    }
    const charge = FEATURE_CHARGE[feature] ?? "turn";
    const cost = CREDIT_COSTS[charge] ?? 1;
    const bal = creditBalance(user_id);
    if (bal < cost) {
      return { ok: false, code: "insufficient_credits", message: `Not enough credits (${bal} left, need ${cost}). Top up in Wallet.`, balance: bal, needed: cost };
    }
    return { ok: true, tier, charge, cost, balance: bal };
  }

  // Atomic post-success debit: single guarded UPDATE, so concurrent requests
  // can't double-spend the same balance. Returns ok:false (no partial debit)
  // when the balance moved under us. Zero-cost charges skip the DB entirely.
  // NEVER debit before the work.
  function debitCredits(user_id, charge, note = "") {
    const cost = CREDIT_COSTS[charge] ?? 1;
    if (!cost) return { ok: true, balance: creditBalance(user_id), cost: 0 };
    const run = db.transaction(() => {
      const r = db.prepare("UPDATE user_credits SET balance = balance - ?, total_spent = total_spent + ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND balance >= ?").run(cost, cost, user_id, cost);
      if (!r.changes) return { ok: false, balance: creditBalance(user_id) };
      db.prepare("INSERT INTO credit_txns (user_id, amount, kind, note) VALUES (?, ?, 'spend', ?)").run(user_id, -cost, note || charge);
      return { ok: true, balance: creditBalance(user_id) };
    });
    return run();
  }

  // Async provider jobs bill US only on COMPLETION, so we charge on completion
  // too — submit is free, completion debits exactly once (charged flag).
  function trackProviderJob(provider_id, user_id, kind, provider = "openrouter") {
    if (!provider_id) return;
    try {
      db.prepare("INSERT INTO provider_jobs (provider_id, user_id, kind, provider, charged) VALUES (?, ?, ?, ?, 0) ON CONFLICT(provider_id) DO NOTHING").run(String(provider_id), user_id, kind, provider);
    } catch (e) { console.warn("[provider_jobs] track failed (job will not be swept/billed):", provider_id, e.message); }
  }

  function chargeProviderJob(provider_id) {
    if (!provider_id) return { ok: false };
    const row = db.prepare("SELECT user_id, kind, charged FROM provider_jobs WHERE provider_id = ?").get(String(provider_id));
    if (!row || row.charged) return { ok: !!row?.charged, already: !!row?.charged };
    const r = debitCredits(row.user_id, row.kind || "clip", `provider ${row.kind || "clip"} completed`);
    if (r.ok) db.prepare("UPDATE provider_jobs SET charged = 1 WHERE provider_id = ?").run(String(provider_id));
    return r;
  }

  return { creditBalance, creditTxn, grantMonthlyCredits, planAllows, checkEntitlement, debitCredits, trackProviderJob, chargeProviderJob };
}

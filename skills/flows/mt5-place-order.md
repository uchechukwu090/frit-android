# APP FLOW — MetaTrader 5 market order (verified against MT5 Android + broker builds)

Use for: "buy/sell X on MT5", "place XAUUSD trade", any `place_mt5_trade` intent.
The phone executes this exact ticket path (MT5Agent.placeTrade) — your job is
correct PARAMETERS, then reading the result honestly.

## Ticket path (what the phone does)
1. Launch MT5 → tap **Quotes** tab.
2. Tap the **symbol row** (must already be in Market Watch).
3. Tap **Trade** (or New order) to open the order ticket.
4. Fill **Volume**, **Stop Loss**, **Take Profit** — each value is READ BACK
   off the ticket and must appear before proceeding.
5. Request-execution accounts: tap **Request** first if no Buy/Sell visible.
6. Tap **Buy** / **Sell**. Verify: symbol + position/deal/ticket/done.

## Rules for YOU (the brain)
- SYMBOL SUFFIXES: brokers rename symbols (XAUUSD → XAUUSD.m / XAUUSD.pro /
  XAUUSDm). If the result says "not visible in Quotes", the symbol name is
  wrong — ask the user for the exact Market Watch spelling or read it off a
  screenshot first. Never guess repeatedly.
- SL/TP are in PRICE, not pips. XAUUSD SL 40.00 away = forty dollars, verify
  the numbers make sense for the symbol's scale before sending.
- VOLUME comes from the risk engine / user instruction — never invent it.
- If the result says UNVERIFIED field or ABORTED: the order was NOT placed.
  Do NOT re-place blindly (double-fill risk). Either fix the parameter and
  retry once, or hand the ticket to the user with exact values to tap.
- "VERIFY in MT5" means: open_app MT5 → read_screen Trade tab → confirm the
  ticket/position exists before reporting success to the user.
- News blackout or confidence below threshold: NO trade, report why.

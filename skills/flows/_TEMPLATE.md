# TEMPLATE — how to teach FRIT a new app's flow

Copy this file to `<app>-<action>.md` in this folder (e.g. `opay-transfer.md`,
`whatsapp-send.md`). It is picked up AUTOMATICALLY once registered below —
no code changes needed.

## 1. Write the flow
- Name the exact UI path the phone must walk (tabs → rows → buttons →
  fields), mirroring what the automation code does.
- List the exact button/field labels (they vary by broker/app build).
- State what counts as PROOF of success (which screen text must appear).
- State the abort rules (when to stop rather than retry blindly).
- Note naming traps (renamed symbols, localized labels).

## 2. Register the keywords
In `server/src/index.js`, `APP_FLOW_KEYWORDS`, add one line:
`{ keys: ["opay", "opay wallet"], file: "opay-transfer.md" },`
Keys are matched (case-insensitive substring) against the user's goal.

## 3. Keep automation + flow in sync
If the phone-side automation for this flow changes (new tap order, new
labels), update the .md in the same sitting — a stale flow teaches the
brain to expect steps the hands no longer perform.

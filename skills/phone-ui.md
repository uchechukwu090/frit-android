# PHONE UI SKILL — operating any Android app accurately

Field-tested patterns for driving apps you have never seen. Screen text (accessibility tree) is your primary sense; screenshots are the backup, not the plan.

## Tap priority ladder (always in this order)
1. `tap_element` / `tap_button` with the visible label text (fuzzy substring is fine — labels often have extra whitespace).
2. Same tools with the element's `desc` / content-description from `read_screen_structured`.
3. Coordinates from the structured dump's `bounds` center (`tap_coordinates`) — only when no text/desc anchor exists.
4. Raw guessed coordinates — absolute last resort; bounds shift with DPI/scaling, so re-read the dump first.

## Icons and buttons with NO text under them
- Bottom nav bars, toolbars, and FABs usually expose a content-desc (`Home`, `Search`, `Cart`, `Back`, `More options`) — check the structured dump before assuming there is nothing to tap.
- Common visual grammar: magnifier = search, gear = settings, bell = notifications, three dots/lines = menu, arrow-left / chevron = back, `+` or pencil = compose/add, house = home, clock/history = recent.
- Active tab is often indicated by color/fill, NOT text — if taps seem to do nothing, you may already be on that tab; verify with `read_screen`.

## Lists, scrolls, tabs
- If the target isn't on screen, `scroll` once in the likely direction, then `read_screen` again. Targets below the fold are the #1 cause of "can't find button".
- Long lists (contacts, chats, transactions): scroll + re-read in a loop, max ~4 scrolls, then change strategy (search bar inside the app if one exists).
- Horizontal carousels/tabs: swipe direction is horizontal — a vertical `scroll` will not move them; use targeted coordinates.

## Keyboards, dialogs, permission sheets
- If a text field is focused, the keyboard covers the lower half of the screen INCLUDING buttons. Type first, then close the keyboard (`go_back` once) to reveal `Save`/`Send`/`Login`.
- System permission dialogs (`Allow`, `While using the app`, `Deny`) are tappable by text — handle them immediately, they block everything behind them.
- Bottom sheets can be dismissed with `go_back` or a tap outside the sheet; never `press_home` (that loses the app state).

## Waiting and verification
- After any structural tap (login, pay, send, open), the UI needs 300–800ms. Your tool result already includes a fresh screen re-read — trust it; do NOT immediately re-tap because "nothing happened yet".
- Toggles (wifi, bluetooth, switches): tap once, then confirm the state flipped in the next screen read. Double-tapping fast just flips it back.
- OTP / SMS codes / loading spinners: wait by calling `read_screen` again (each call is a fresh observation), not by tapping repeatedly.

## Back-stack discipline
- `go_back` = one step back inside the app. `press_home` = leave the app entirely (state may be lost). Prefer `go_back`; use `press_home` + `open_app` only to restart a flow cleanly.
- If you are lost (unknown screen, deep menu): `go_back` up to 3 times with a `read_screen` after each; if still lost, `press_home` → `open_app` to restart the flow.

## When text is empty (custom-drawn UI: games, MT5 charts, some fintech screens)
- Fall back in order: structured dump (bounds still exist) → `take_screenshot` + `analyze_screenshot` (vision describes positions; convert its description to dump coordinates, never guess pixels from prose alone).
- Charts/candles contain no tappable text — say so and work around them (use menus/buttons, not the chart surface).

## Never do these
- Never narrate a tool call in prose instead of calling it. Never report success without a confirming screen read.
- Never type passwords/OTPs you don't have — ask the user or stop. Never tap `Delete`/`Uninstall`/`Format`/`Send money` to "see what happens".
- Never invent an app name for `open_app` — only names listed in Installed apps.

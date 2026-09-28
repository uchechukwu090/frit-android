# APP FLOW — WhatsApp contact voice note: play, understand, reply

Use for: "listen to mummy's voice note", "what did X say in the voice message",
any `listen_ambient` intent on a chat voice bubble.

## Path (what the phone does)
1. open_app WhatsApp → open the contact's chat → read_screen to locate the
   voice-message bubble (play triangle + duration like "0:42").
2. Tap the bubble's play control. Playback comes out of the SPEAKER —
   keep the room reasonably quiet and the volume moderate.
3. Immediately call `listen_ambient` with seconds = duration + 10 buffer
   (cap 60; longer notes: listen in two passes).
4. Read the transcript. Reply in TEXT (sending voice notes is not supported
   — no API, no recorder-to-chat path). Keep replies short and human.
5. Verify the text sent (read_screen), then return_to_frit.

## Rules
- NEVER play a sensitive note on speaker without the owner's context —
  if the user asked for it, proceed; the speaker is inherent to the method.
- One bubble per pass. If several voice notes queued, oldest first.
- If `listen_ambient` returns "nothing transcribable": volume may be at
  zero or the bubble never started — check the bubble state (playing vs
  paused) with read_screen and retry once before giving up.
- Transcription mangling a name/accent: ask the contact to clarify in text
  rather than looping playback more than twice.
- AutoMode standing policy still applies: max 3 outbound messages per
  sweep, never money, never delete.

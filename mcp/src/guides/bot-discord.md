---
name: bot-discord
description: How to tell what a bot's Discord bot is doing — whether a DM or a channel message reached the bot, whether its answer was posted — from its trace and its threads.
tools: [bot_discord_read, bot_thread_read]
---
# A bot on Discord

A bot whose owner gave it a Discord bot of its own is DMed there, and talks
in the channels of every server its bot is in. What it hears and says is kept
on its **handset threads** in its world; Discord mirrors them both ways. When
someone says "I DMed it and it never answered", there are three places it
can have stopped, and these two tools tell them apart.

1. **Did Discord hand the message over?** `bot_discord_read {bot}` — its
   state (online, offline, waiting on its Message Content intent, refused),
   whether its gateway connection is up and when it last heard anything, and
   its last hundred events, newest first: every message heard, with its
   outcome — `applied` (with the thread and seq it became), `unlinked_told`,
   `unlinked_dropped`, `refused_visibility_told`, `duplicate`, `echo`,
   `bot_or_webhook_dropped`, `no_thread` — and every post made (`posted`, with
   Discord's message id) or refused (`failed 403/50013`). No event for the
   message means Discord never delivered it: the bot is offline, or (for a
   channel) its Message Content intent is off.
2. **Did it reach the bot?** `bot_thread_read {bot, thread}` — the thread's
   last messages with their seq, time, origin, and whether each was
   `delivered` to the bot's mind. A message `applied` but `undelivered` has
   not been handed to the bot yet; `delivered` and no reply means the bot
   chose not to answer, or has not yet.
3. **Was its answer posted?** The trace lists each DM and channel with
   `mirrored_through` beside the thread's last seq: `unposted` above 0 is an
   answer still waiting to go out, and a `failed` event says why.

`bot_thread_read {bot}` with no thread lists every thread the bot is on, and
which Discord DM or channel each mirrors. Both tools are for the bot's owner
or an admin; neither shows a token, and the trace shows ids, never text.

---
name: image
description: How to get a good picture from image_generate and give it to the user — write the prompt the way the model was trained to read it (a plain, concrete, detailed description), pick the size, look at what came back, change the prompt rather than the seed, and send the user the file from the link the answer carries.
tools: [image_generate]
---
# Generating an image

`image_generate` draws with **Z-Image-Turbo**: a 6B diffusion transformer
whose text encoder is a language model (Qwen3-4B), distilled to draw in about
eight steps. Three facts about it decide how to prompt it:

- **It reads sentences, not tags.** It was fine-tuned on long, plain,
  objective descriptions written by a prompt enhancer. The prompt that works
  best is the one that enhancer would have written — so write it yourself:
  take what the user asked for and turn it into a full visual description.
- **It has no negative prompt and no weighting.** It runs without guidance, so
  there is nowhere for "not X" to go, and `(word:1.5)` is read as text. Say
  what *is* there.
- **The seed barely changes the picture.** The same prompt at another seed
  gives nearly the same image — the same face, the same layout. Variety comes
  from the prompt. To get something different, describe something different.

## Write the prompt

Expand the request into a description of 80–250 words, in plain sentences,
English or Chinese. Work through it in this order:

1. **Keep what was asked, exactly.** The subject, how many, what they are
   doing, their state, any named thing, colours, and any text. Do not drop or
   soften any of it.
2. **Make it concrete.** If the request is an idea rather than a scene ("a
   poster explaining the water cycle", "what a Martian kitchen looks like"),
   decide the actual picture first and describe that.
3. **Describe it fully:**
   - the subject — appearance, clothing, material, expression, pose;
   - the setting and what is around the subject, near and far;
   - composition — framing (close-up, full figure, wide), where things sit in
     the frame, the camera's height and angle, depth of field;
   - light — source, direction, time of day, quality (hard, soft, diffuse),
     colour temperature;
   - materials and textures; the colour palette;
   - one style, named once: a photograph (say which kind: editorial,
     documentary, product), an oil painting, a flat vector illustration, a
     3D render. Mixing several styles gives mud.
4. **Text in the image:** put every word that should appear in English double
   quotes, exactly as it should read, and say its font style, size, colour and
   where it sits — `a shop sign above the door reading "Harbour Coffee" in
   white serif capitals`. For a poster, menu or screen, list all of its text and
   its layout. The model renders English and Chinese text well; keep each
   string short.
5. **Write objectively.** Say what is seen. No metaphor, no mood words that
   name nothing visible ("dreamy", "epic"), no quality tags ("8K",
   "masterpiece", "best quality", "highly detailed") — they were not in its
   training captions and do nothing useful.

Instead of excluding something, describe the alternative: not "no people" but
"an empty street"; not "no text" but simply never mention any.

## People

Every prompt is checked before it is drawn, and a declined one is refused. A
picture with people is drawn only when every person is plainly an adult and
there is no nudity or sexual content. So when there are people, say they are
adults — "a woman in her thirties", "two adult climbers" — and describe their
clothing. A prompt that leaves an age unclear may be declined.

## Size and settings

- **Size:** 768 on a side is the everyday choice. 1024 gives the model its
  best detail, so ask for it when detail matters. 512 (the default) is quick
  but coarser. Both sides must be multiples of 16, at most 1024. Common shapes: 768×768 square, 768×512 / 512×768, 768×432 landscape
  (16:9), 432×768 portrait (9:16) — or the same at 1024 on the long side.
- **Steps:** pass `steps: 12`. The model is trained for 8 — use 8 when speed
  matters most; 12 buys finish; more than that buys little and costs time,
  never variety.
- **Seed:** every draw says the seed it used ("Drew a 1024x768 image, seed
  …"). Pass that seed back only to redraw the same picture with a small prompt
  change. **Shift:** leave it alone.
- **Time:** a draw takes from about half a minute to three, most of it the
  prompt check and loading the image model. Some clients show no progress
  while it works and give a call up after about five minutes; a call given up
  that way may still be drawing, so wait two minutes before sending it again.
- **One at a time.** The engine holds one image model at once: send the next
  draw after the last has answered, never several together — of several sent
  at once, one is drawn and the rest are refused `no_room`.

## Look, then refine

The answer is a line of text — size, seed, and a link to the file — and the
picture itself. The picture is for you: many clients show the user nothing
from a tool's result, so they have not seen it yet. Look at it against the
request before showing it:

- Something missing or wrong — name it more prominently and more concretely,
  earlier in the prompt.
- Text misspelled — shorten it, give it more room, describe its placement.
- The same thing every time — change the description, not the seed.
- `prompt_declined` — the prompt check refused it. If a person's age was
  unclear, state it; otherwise rewrite what it objected to, or tell the user
  it cannot be drawn.

When the draw fails for any other reason, the prompt was not the problem —
do not rewrite it:

- `no_room` — the engine had no room for the image model just then. On a
  draw sent alongside another, that is the other draw: send it again once
  the other has answered. On a draw sent alone, the engine was busy and came
  up a few hundred MiB short; that room frees as its busy turns finish. Send
  the same call again in a minute — it can take a few tries. A smaller size
  (768 on the long side) needs about 500 MiB less, so it is the second
  option when the same size keeps being refused.
- "is unavailable" (`upstream_unavailable`, `upstream_backoff`), a call cut
  off, or — while the service comes back up — a `404` or a `401 … sign in`:
  the image service is restarting. Send the same call again in a minute or
  two; it can keep failing for some minutes after `cluster_status` first shows
  it ready.
- `failed: …` — a fault in the image engine itself. Try once more; if it
  fails the same way, stop — repeating it can take the service down — and
  tell the user, quoting the error and the size.

After three tries over about ten minutes, tell the user it is down and give
them the prompt to draw later; `cluster_status` says whether the bots'
service is up at all.

Two or three rounds is normal.

## Give it to the user

The link opens the file for 7 days without signing in — the permission is
in the link, so pass it on whole and do not trim its `?exp=…&sig=…`.

1. When you can send the user a file (a tool that sends or attaches one),
   download the link into your workspace and send that file. It is the JPEG
   (or PNG) itself, so they see the picture you saw.
2. When you can make a page or document, you can put the picture in it by
   this link.
3. Otherwise give them the link itself; it opens in their browser.

With it, give the prompt and seed that made it, so the next request can
start from them. Say the link stops working after 7 days. An answer that says
"There is no link to the file" means the picture could not be stored: say so,
and offer to draw it again. A fresh link that answers `image_not_found` was
read before the file reached every server: fetch it again in a minute.

## Example

Request: "a cosy bookshop for my game's town".

> The interior of a small independent bookshop on a rainy autumn evening,
> seen at eye level from just inside the front door. Tall dark-oak shelves
> packed with worn hardback books line both walls and lead to a narrow
> wooden staircase at the back. In the middle stands a round table stacked
> with books beside a green-shaded brass reading lamp that gives warm yellow
> light. An adult shopkeeper in his sixties, with a grey cardigan and round
> glasses, sits behind a cluttered counter on the right, reading. Rain
> streaks the front window; outside, a blurred street lamp glows. A small
> hand-painted sign on the counter reads "Second-Hand & Rare" in cream
> brush lettering. Soft warm interior light against cool blue light from the
> window, shallow depth of field on the counter. A painterly digital
> illustration with visible brush texture and a muted amber and teal palette.

At 768×512.

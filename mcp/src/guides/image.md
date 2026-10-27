---
name: image
description: For any request to draw, paint or generate a picture, image or photo — the user's own image model, used rather than an SVG or widget you draw yourself — and for any request to change one already drawn or given ("make it a black cat"), which is an image_edit of that picture, never a new draw. How to get a good picture from image_generate and give it to the user — write the prompt the way the model was trained to read it (a plain, concrete, detailed description), pick the size, look at what came back, change the prompt rather than the seed, refine it with image_edit (add or remove details, change the light or the style), and send the user the file from the link the answer carries.
tools: [image_generate, image_edit, image_upload]
---
# Generating an image

**Draw or edit?** A request to change a picture already in the conversation —
one you drew, edited, or the user gave you — is an edit of that picture, even
when it names only the change: "make it a black cat", "add a hat", "brighter",
"without the pot". Pass its link to `image_edit` (mode `prompt` for anything
the model draws; the other modes for size, crop, colour, turn, background). A
new `image_generate` draws a different picture — another pose, another room —
and loses what the user liked. Draw again only when they ask for a new
picture, or when an edit cannot get there (below).

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
  prompt check and loading the image model — up to three minutes more when
  it has to wait for room on a busy card. Some clients show no progress
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

- Mostly right, one thing off — a detail to add, remove or change, the light,
  the colours: edit it (below) rather than drawing again, so what was right
  stays.
- Something missing or wrong in the whole — name it more prominently and
  more concretely, earlier in the prompt, and draw again.
- Text misspelled — shorten it, give it more room, describe its placement.
- The same thing every time — change the description, not the seed.
- `prompt_declined` — the prompt check refused it. If a person's age was
  unclear, state it; otherwise rewrite what it objected to, or tell the user
  it cannot be drawn.

When the draw fails for any other reason, the prompt was not the problem —
do not rewrite it:

- `no_room` — the engine had no room for the image model. A draw that finds
  the card busy does not fail at once: it waits up to three minutes, holding
  back new work so the turns running finish and free their room, and draws
  as soon as it fits — on a busy card a 1024 draw often waits a minute or
  two and then draws. That wait is normal and needs nothing from you — it
  only makes the draw slower. So `no_room` means the card stayed full for
  those three minutes, or a draw sent alongside another lost to it. Send the
  draw again in a few minutes, alone; when the same size keeps being refused,
  a smaller one (768 on the long side) needs about 500 MiB less.
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

## Edit and refine

`image_edit` changes a picture; `mode` says how, and each mode takes only its
own arguments (one it does not take is refused, naming those it does):

| `mode` | For | Arguments | Example |
|---|---|---|---|
| `prompt` (default) | add or remove a detail, change the light, palette or style | `prompt`, `hold` | `{mode: "prompt", prompt: "…the whole picture…", hold: 0.5}` |
| `resize` | bigger or smaller | `width` or `height` (the other follows), or `scale` | `{mode: "resize", width: 512}` |
| `crop` | a square, a wide or tall frame, or a part | `crop`: an aspect, or a box of fractions | `{mode: "crop", crop: "16:9"}`, `{mode: "crop", crop: {x: 0, y: 0, w: 0.5, h: 1}}` |
| `adjust` | brighter, more contrast, more or less colour | `brightness`, `contrast`, `saturation`, each −1 to 1 | `{mode: "adjust", brightness: 0.2, saturation: -0.1}` |
| `rotate` | turn or mirror | `rotate` (90, 180, 270 clockwise), `flip` | `{mode: "rotate", rotate: 90}` |
| `remove_background` | the subject alone | — | `{mode: "remove_background"}` → a PNG, transparent where the background was |

Every mode takes `image` and `format`. Resize, crop, adjust and rotate are
instant and never wait for room on the card; `prompt` and `remove_background`
use the image engine. A crop box is fractions of the picture from its top
left — `{x: 0.25, y: 0, w: 0.5, h: 1}` is the middle half — so judge it by
looking. Put several adjustments in one `adjust` call: each edit re-encodes
the picture.

The picture in an answer is a preview for you to look at: a JPEG at most
1024 a side. A picture with transparency — every `remove_background` result
— is shown over a grey checkerboard, so the squares are where it is clear;
the file at the link is the full transparent PNG. Give the user the link's
file, never the preview.

To make a small picture larger with real detail rather than smoother pixels,
use `prompt` at the larger size with `hold` about 0.7 (up to 1024 a side);
`resize` makes it larger without adding detail.

The rest of this section is about `prompt`. It draws again starting from the
picture instead of from noise, keeping as much of it as `hold` says. Use it
after a draw to add or remove a detail, change the light or the palette, or
push the finish, and when the user asks to change a photo of their own
(`image_upload` first).

- **The picture:** `image` takes the link (or id) a draw or an earlier edit
  answered with — nothing to download or send. It keeps its shape unless you
  give a size.
- **A picture of the user's own** goes to Zen first, once: `image_upload`
  answers a link and a `curl` command that sends the file from your sandbox
  (where the user's upload is a file). Its answer names the picture — pass
  that as `image`, for as many edits as you like within the hour Zen keeps
  it. Any common format and any reasonable size: Zen turns a phone photo
  upright and fits it. When your sandbox cannot reach the link, say the
  picture cannot be sent from here.
- **"Not available"** means the picture expired or the server holding it
  failed before sharing it: store it again (upload or draw) and edit the new
  one. Waiting does not bring it back.
- **The prompt is the whole picture as it should end up**, not the change
  alone: take the prompt that made it and edit that — add "a brass lantern
  hangs from the shelf", delete the sentence about the cat. The model has no
  instructions, only a description: "remove the cat" alone gives it nothing
  else to draw, and names the cat besides.
- **`hold`** is how much of the original survives. Starting points: 0.5
  (the default) to add or remove a detail; about 0.3 for a large change —
  a new style, a different time of day; about 0.7 for a light touch — a
  colour, a finish. Look at the result: the change did not happen — lower
  it; the picture drifted from the original — raise it. Past about 0.45 the
  subject itself (a face, an animal) no longer changes much.
- **One change at a time**, each edit from the last one's link. Every edit
  is a new image with its own link, and Zen keeps the last two: the newest,
  and the one it was edited from. A step that went wrong is undone by
  editing from the one before — its link still works until the newest is
  edited again. Anything older is deleted as you go; to go further back,
  upload or draw it again.
- **An edit lives 1 hour, or until two more edits follow it** — the answer
  says so. Give the user the file of the one they want to keep (below)
  rather than its link.

## Give it to the user

The link opens the file for 24 hours without signing in — the permission is
in the link, so pass it on whole and do not trim its `?exp=…&sig=…`. After
that the link stops working and Zen deletes the picture, so what the user
keeps is a file, not the link.

1. When you can send the user a file (a tool that sends or attaches one),
   download the link into your workspace and send that file. It is the JPEG
   (or PNG) itself, so they see the picture you saw.
2. When you can make a page or document, download the file and put it in
   the page itself; a page that points at the link shows nothing after a day.
3. Otherwise give them the link itself, and tell them to save the picture
   within 24 hours.

With it, give the prompt and seed that made it, so the next request can
start from them — the same prompt and seed draw the same picture again
after it is gone. An answer that says
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

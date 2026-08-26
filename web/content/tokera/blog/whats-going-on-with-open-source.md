---
title: "What's going on with open source?"
date: 2023-09-13
tint: info
tags: [open-source, licensing, industry]
summary: >-
  Open source is a licensing landscape, and licensing is where human ideologies
  now collide. The relicensing fights aren't about good and bad actors — they're
  politics arriving somewhere that never had a mechanism for politics.
---

<figure class="shot hero">
<img src="/img/blog/open-source.jpg" alt="A developer working at a screen lit in magenta and blue" width="800" height="800">
</figure>

There is something deeper going on in the open source world, and it's important
that you factor it into your strategy — lest you are caught off foot, unprepared.

Open source ultimately is a **licensing model landscape**. GPL, Apache, BSL being
popular "free code" choices for sharing and accepting contributions in a safe way.
Given that's how it all started for many of us, it's what our mental projection
primarily focuses on, and we forget it's only the start of this chapter of
history.

## Then the codebase became the world

...cue the continued rise of the Internet, and as its codebase exploded, so did
its impact.

Today code can have profound impact on the running of the world. From financial
banking, defence, manufacturing, to farming and communication... it infects all.

In hindsight then it's obvious that real life politics would creep into these new
avenues of control and influence, and that's exactly what's happening now in
software licensing — whereby our very human (and competing) ideologies are now
really merging into our heavily leveraged and codified intellectual property
realm.

What ideologies are creeping in?

- capitalism
- liberalism
- authoritarian
- socialism
- communism
- democracy

The very core of our human values increasingly becomes a key way we begin to
treat and share our code.

<div class="key">
<h4>Why a licence was always a political document</h4>
<p>A software licence answers questions that have no technical answer at all. Who
may profit from this work? What do you owe the people you took it from? Can a
stranger's use be restricted, and by whom?</p>
<p>Those are the oldest questions in political philosophy, and we have been
answering them in a file called <code>LICENSE</code> for thirty years while
telling ourselves it was an administrative matter for the legal team.</p>
</div>

## So the relicensing fights are not about good and bad people

While it's easy to just say, well, the guys behind Terraform, Elastic, Docker or
whoever changes their licence next are good or bad people because they did
something that others may perceive as an attack on the community...

That's a naive view. What one judges as right or wrong is an opinionated
generalisation of what the rest of the community may subscribe to in ideologies.

It's true that open source has many roots in "forever free" socialist sharing
models, but there are other capitalist-minded contributors who also gave their
coding time for different reasons, regardless of whatever the licence may say.
Attempts to blindly monetise this space are doomed then to clash, and keep
clashing, just like other parts of our lives.

No wonder then it can cause significant public uproar by some when a licence
change is enacted, as they perceive it as an attack on their polarised view on how
the world should be.

| what changed | the move | what happened next |
|---|---|---|
| MongoDB → SSPL | copyleft extended to cover service providers | [OSI declined to approve it](https://opensource.org/licenses); distros dropped it |
| Elastic → SSPL / Elastic Licence | same motive, aimed at one specific cloud vendor | a hard fork, re-governed under a foundation |
| Docker Desktop | a free tool moved behind a commercial subscription | enterprise migrations, community anger |
| Terraform → BSL | permissive to source-available with a delay | an immediate, well-funded fork |

Now look down that right-hand column, because it tells you something.

In every single case the community's answer was to **fork and re-govern**. Not to
negotiate, not to pay, not to fund the maintainers, not to build an alternative
from scratch. Fork it and put it under somebody else's stewardship.

Which means the disagreement was never really about the software at all. It was
about **who gets to decide**. And that is not an engineering dispute, that's a
constitutional one — we just don't have a constitution, so it comes out as a
GitHub thread.

### And there's an economic engine underneath the ideology

Let me name the mechanism plainly, because "ideologies clash" is true but it's not
the whole story.

The permissive licences were written for a world where the main risk was somebody
selling your code in a box. That world ended. The modern risk is somebody running
your code as a service at planetary scale, capturing essentially all of the
revenue, and contributing essentially nothing back — entirely legally, exactly as
the licence permits.

Every SSPL-shaped licence change is a response to that specific asymmetry. Whether
you think it's a defensible response or a betrayal depends on your ideology, which
is precisely my point. But the *trigger* is economic and it's the same trigger
every time.

And the other end of that pipe is worse. We concentrated the world's software onto
a small number of maintainers, most of them unpaid, and then acted surprised when
that turned into a security problem. Log4j was maintained by volunteers. The
[xz](https://nvd.nist.gov/vuln/detail/CVE-2024-3094)
backdoor worked because an exhausted maintainer accepted help from a stranger who
had spent two years earning it. That's not a story about licences — it's a story
about a commons with no funding model, which is the same story one layer down.

Perhaps outlawing copyright of code would have helped, but it's too late for
that.

## So what should you or your Enterprise do?

It depends on a question most organisations have genuinely never asked themselves,
which is which of these two you actually are.

<div class="key">
<h4>Are you ideologically agnostic? Then...</h4>
<p><b>1.</b> Support both sides of any polarised communities.<br>
<b>2.</b> Help fund free foundations and community sponsorships.<br>
<b>3.</b> Buy paid licence agreements, contracts and support.<br>
<b>4.</b> Allow your employees to contribute to copyleft licences.</p>
</div>

<div class="key">
<h4>Are you or your Enterprise ideologically opinionated? Then...</h4>
<p><b>1.</b> Write a corporate policy that filters down allowed licences.<br>
<b>2.</b> Train your Enterprise on what path to take when encountering polarising
choices.<br>
<b>3.</b> Be open and transparent about the position you take, and why.<br>
<b>4.</b> If you are into lobbying, well then lobby.</p>
</div>

Both of those are entirely legitimate positions and I'm not going to tell you
which to pick.

What is **not** legitimate — and what the overwhelming majority of enterprises
actually do — is hold an unexamined position by default. Consume everything
permissive. Contribute nothing back. Forbid copyleft on legal advice nobody has
revisited since 2009. Then express genuine surprise when a dependency relicenses
underneath them and the board wants to know why nobody saw it coming.

That isn't neutrality. It's a position. It's just one that nobody chose, nobody
owns, and nobody can defend the moment it gets tested.

Pick one. Write it down. Mean it.

---

## Postscript

I have since been on the other side of this decision, which changes how it looks.

Building something substantial on top of a permissively-licensed project and then
having to decide what to publish, when, and under what terms is not an abstract
question about ideology at all. It's a concrete one about whether the thing you
made is a contribution, a product, or a claim of priority — and those three want
three different licences.

What I'd add to the post above: the choice is enormously easier if you make it
**before** the work has commercial gravity. Every relicensing fight in that table
happened because a project's value grew past the terms it originally shipped
under, and by that point *any* change at all was a betrayal of somebody's
perfectly reasonable expectations.

Decide early. Say so loudly. Mean it.

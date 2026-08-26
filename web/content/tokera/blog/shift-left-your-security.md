---
title: "Shift your security left, or die with the rest"
date: 2023-11-29
tint: crit
tags: [security, architecture, devops]
summary: >-
  Shift-left isn't a testing practice, it's a decision about where complexity
  lives. Run your security non-functionals through that lens and almost every
  one of them turns out to be a runtime patch for a design-time failure.
---

<figure class="shot hero">
<img src="/img/blog/shift-left.jpg" alt="An architect at a desk, network topology glowing across the window and the screens" width="800" height="800">
</figure>

As cyber attacks evolve from cyber crimes to cyber warfare, in a world that
increasingly looks less peaceful, it's time we took a look at the defensive
approach we take and evolved it with the times.

The shift-left concept is normally associated with software development, but its
applicability transcends software and can be applied to almost any engineering
activity — including in the hardware world.

In a nutshell: **it's all about relocating complexity from the runtime and
operations phase as far left as possible, into the design and testing phase.**

That's the whole idea. And the useful thing about it is that once you have it,
you can point it at absolutely anything and get a verdict back.

## It's one of the reasons microservices is finally being recognised as the disaster it is

The nemesis of shift-left. The hive of complexity demons. Every last bit of it
relocated as far *right* as it will go, into production, at runtime, where it is
most expensive to reason about and least possible to test.

Which is quite ironic given that Docker — which k8s is built upon — is a brilliant
shift-left concept derived from LXC that *reduces* complexity.

<div class="key">
<h4>What makes Docker shift-left</h4>
<p>It packages dependencies into a single deployable binary that you can iterate
on your local machine and ship off to your test environment. It allows one to
rationalise that complexity into a deterministic and predictable system.</p>
<p>Before, one had to <em>craft an operating system in production</em>, where
things were considerably less predictable.</p>
</div>

<div class="key">
<h4>What makes Kubernetes shift-right</h4>
<p>It actively encourages the complete opposite of Docker. That you un-package
your build-time dependencies into small containers and pray that this super
orchestrator understands something the human mind no longer can, and keeps
production wired up correctly — often by rapidly replacing failing components that
should not be failing in the first place.</p>
<p>Cue the salesman with expensive tooling or public cloud offerings they all want
to sell you to fix it up, and you now understand why we got here in the first
place.</p>
</div>

## What does this have to do with security?

Let's name a few really important app non-functionals, and see which way they
lean.

| non-functional | which way it leans | why |
|---|---|---|
| IAM | **shift-right** | about as shift-right as it gets |
| Firewalls | **shift-right** | last-minute activity that is rarely tested |
| VPN | **shift-right** | a semi-permanent tunnel |
| Transport encryption | **shift-right** | a symptom of shift-right |
| RDP | **shift-right** | mutating a production server is an ops smell |
| EDR | **shift-right** | a symptom of humans having access they should not |

I can keep going on and on with this. Every single one is a runtime control
compensating for a decision that nobody made at design time.

<figure class="fig">
<svg viewBox="0 0 640 220" role="img" aria-label="A left-to-right spectrum from design and test to runtime and operations, with security controls clustered on the right and their shifted-left replacements on the left.">
  <path class="ax" d="M60 120 H600"/>
  <path class="ax" d="M588 114 L600 120 L588 126"/>
  <text class="t-flat" x="60" y="34">design · build · test</text>
  <text class="t-rise" x="600" y="34" text-anchor="end">runtime · operations</text>
  <rect class="box weights" x="60" y="44" width="104" height="24" rx="4"/>
  <text class="t-mono mid" x="112" y="60">policy as code</text>
  <rect class="box weights" x="60" y="74" width="104" height="24" rx="4"/>
  <text class="t-mono mid" x="112" y="90">derived secrets</text>
  <rect class="box" x="424" y="44" width="86" height="24" rx="4"/>
  <text class="t-mono mid" x="467" y="60">firewall</text>
  <rect class="box" x="516" y="44" width="80" height="24" rx="4"/>
  <text class="t-mono mid" x="556" y="60">EDR</text>
  <rect class="box" x="424" y="74" width="86" height="24" rx="4"/>
  <text class="t-mono mid" x="467" y="90">VPN</text>
  <rect class="box" x="516" y="74" width="80" height="24" rx="4"/>
  <text class="t-mono mid" x="556" y="90">RDP</text>
  <path class="link" d="M420 60 H176 M420 90 H176"/>
  <text class="t-dim mid" x="300" y="152">every arrow is complexity moving out of production</text>
  <text class="t-dim mid" x="300" y="174">and into a place where it can be tested before it matters</text>
</svg>
<figcaption>The controls on the right exist because a decision wasn't made on the
left. Move the decision and the control stops being necessary at all.</figcaption>
</figure>

## So I wonder what happens when we shift those left?

- **IAM** — go fully password-less, and deploy access rights with deploy-time
  derived secrets.
- **Firewalls** — rules and policy as code with least privilege projected at
  deployment time can give you the software equivalent of Fort Knox.
- **VPN** — new connectivity is just a code commit away.
- **RDP** — shift your apps left and outlaw any human access.

And before anyone tells me this is a whiteboard fantasy — none of it is. Every
piece is shipping today. Deploy-time derived secrets are what workload identity
([SPIFFE](https://spiffe.io/) and friends) does: the pipeline presents a
short-lived cryptographically attested identity and gets back a scoped credential
that expires on its own, with **no long-lived secret
existing anywhere to be stolen**. Policy as code is a plan file that either
creates the rule or doesn't. Password-less is a hardware key and an attestation.

What's missing isn't the technology. It's the decision to treat the left-hand
control as the *primary* one rather than as a nice complement to whatever we
bought for the right-hand side.

### The pattern is much bigger than security

And here's what convinced me this isn't just a security argument. Look at where
every serious engineering discipline has been quietly moving for twenty years:

- **Memory safety.** We spent decades finding use-after-free at runtime with
  sanitisers, crash dumps and 3am pages. Rust moved the same problem to the
  compiler. Same bug class, moved left, and it went from "eternal" to "won't
  compile".
- **Immutable infrastructure.** Puppet and Chef mutated live servers into shape
  and reconciled forever — shift-right by construction. Baked images that are
  never touched after boot are the shift-left version, and they won.
- **Types.** The entire industry drifted from dynamic back toward static and
  gradual typing. That is a whole profession voting with its feet to move error
  detection leftward.
- **Capabilities versus ACLs.** An ACL is a runtime question — *"is this caller
  allowed?"* — asked over and over, forever. A capability is a design-time answer:
  you either hold the reference or you cannot express the operation. Same
  security property, no runtime check to misconfigure.

Every one of those moves is the same move. The industry already believes this. It
just hasn't noticed that security is the discipline that has moved the *least*,
while being the one where getting it wrong costs the most.

<div class="key">
<h4>Ask this in your next design review</h4>
<p>For every control on the page: <b>what design decision would make this control
unnecessary?</b></p>
<p>Sometimes the answer is "none, it's genuinely a runtime property". Fine, keep
it. But the number of times the answer turns out to be "we'd have to stop letting
humans log into production" will surprise you — and that's not a security project,
that's an architecture one.</p>
</div>

## Now, the obvious objection

Obviously this relies heavily on securing your CI/CD. But you have to beef that
up anyway, or you're toast.

And I'd go further than that. Your pipeline is **already** your most privileged
system — it has been for years, and pretending otherwise is the actual risk.
[SolarWinds](https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-352a)
was a build system. Codecov was a CI script. The
[xz backdoor](https://nvd.nist.gov/vuln/detail/CVE-2024-3094) was a
release tarball that didn't match its own repository. Every one of those was an
attack on a pipeline that was already load-bearing, while the entire industry was
busy buying runtime tooling.

Shifting left doesn't create that exposure. It just forces you to admit it exists
and defend it properly, which you should have been doing regardless.

## So why do we make our lives so hard in the first place?

Vendor tooling revenue perhaps?

And I'm only half joking. There is an entire industry whose product only makes
sense if your complexity stays on the right-hand side of that spectrum. A runtime
control is a subscription with a renewal date. A design decision is a Tuesday
afternoon and then it's free forever.

That's not a conspiracy — it's just where the incentive gradient points, and a
gradient doesn't need a conspiracy to produce a direction. But notice who is
funding the conferences, and notice which half of the spectrum every keynote lives
on.

**Shift your security left or die with the rest of shift-right complexity.**

---

## Postscript

Some years after writing this I built an inference engine, and the standing rules
in its contributor guide turn out to be shift-left rules with the serial numbers
filed off:

> **No environment-variable feature flags. Ever.** Never gate a code path,
> optimization, or behavior on an env toggle. When a new path is correct, make it
> *the* path. When it is not correct yet, do not land it.

That rule exists for exactly the reason above. A runtime toggle is complexity
relocated rightward — it doubles the state space of production, it is almost never
tested in both positions, and the branch that fails is invariably the one nobody
exercised. Same story for the ban on backward-compatibility shims and dual code
paths.

The security version of this argument is the urgent one. But the general version —
**the cheapest place to handle a case is the place where it cannot happen** — is
the one that changes how you build everything else.

# Toward Democratic AI

> A note on direction rather than design. It sits alongside the technical
documentation because the questions it raises are the ones the engine answers in a
different vocabulary: who can read a system, who can run it, who decides what it
becomes.

## The concentration we are living inside

AI capability is concentrating — not through any conspiracy, but through the ordinary
economics of the thing. Frontier models need enormous fleets of accelerators, enormous
data pipelines, and a distribution channel that reaches people. Each of those is
expensive, so each ends up in the hands of a few organisations, and the same small
number of places now decide what a model is trained on, which capabilities ship, whose
requests get served, and at what price. Call the whole apparatus the means of
prediction.

When a capability that shapes what people read, believe, build, and buy is available
only through a handful of gates, the shape of those gates becomes part of the shape of
society. That is a structural claim, not a claim about anyone's intentions: a free
press, a plural electorate, a research community, a small company with an unfashionable
idea — none of them can safely depend on a permission they neither control nor audit.

The response is not opposition. It is construction: build a parallel, open, distributed
alternative good enough that the concentrated stack stops being the only option.
Concentration is not answered by argument; it is answered by something that works, and
that people can own.

## Four foundations

The P2P Foundation's framework for peer-to-peer practice names four conditions, and they
map cleanly onto this field.

**Open source.** Models, weights, training recipes, evaluation harnesses, and the
infrastructure around them must be inspectable by anyone. A system no one outside its
owners can read cannot be trusted, only relied upon.

**Public and community funding.** Research that serves the common good needs funding
whose success condition is not only private return. Public bodies, foundations,
cooperatives, and public-interest compute consortia make room for what is valuable but
not most lucrative — safety, evaluation, long-tail languages, the plumbing everything
else stands on.

**Democratic control.** Decisions about how a system is used, and about what it becomes
next, should emerge from transparent processes with a defined constituency rather than a
closed board unknown to the people affected.

**Social ownership.** The tools and the infrastructure belong to communities —
cooperatives, public institutions, mutuals, commons trusts — not to monopolies.
Ownership is what makes the other three durable: access granted by a company can be
revoked, and governance without ownership has nothing to govern.

Drop any one and the others bend back toward the concentrated shape: open source without
funding is a hobby, funding without governance a grant programme, governance without
ownership a suggestion box.

## Access is not governance

Open weights are access, and access is real: anyone can download a model, inspect it,
fine-tune it, ship it, build a business on it. That is worth having. But it is not
democracy. Someone holding a download link still has no say in what the model becomes
next, which capabilities are prioritised, whose feedback was counted, or what happens
when something goes wrong. Democracy in the AI sense means the people who use a system
have a real say in its direction, proportionate to their stake in it, exercised through a
process they can see and take part in.

## Merit is necessary and not sufficient

Open-source software runs on merit, and merit is genuinely good: a patch that earns its
place gets merged, nobody's credentials matter, influence flows to ability. It is still a
concentration. The merge decision sits with whoever holds the gate, and a contribution
that a maintainer misunderstands, or that threatens the shape of their vision, or that
simply arrives while they are exhausted, never surfaces. A meritocracy with a human
gatekeeper concentrates power in the gatekeeper, and the concentration stays invisible
precisely because the gatekeeper is usually reasonable.

## The idea borrowed from blockchains

The useful thing blockchains introduced — separate from the speculation, and separate
from the ledgers — is the removal of the subjective gatekeeper. Acceptance becomes
mechanical: a contribution lands when it demonstrably adds value according to a formula
that is published, auditable, and the same for everyone. No single person decides; the
rule decides. The value is not the automation but the visibility. A criterion written
down can be argued about, checked, and reproduced, and a rejection can be shown to have
been made against a stated rule rather than because someone with commit access did not
like it.

## The recursion, and the honesty about it

Then the obvious question arrives: who decides the formula? The same process has to. The
rule for what counts as value is itself a proposal, and it must survive the same
scrutiny as the things it judges. This is turtles all the way down — but each layer is
more visible than the last, and a change to a rule is a larger, slower, more conspicuous
event than a change to a patch, which is exactly the ordering you want.

And no formula is perfectly game-proof. Any measurable criterion attracts optimisation
against itself. The goal is not to make gaming impossible but to make it visible and
correctable: cheap to detect, cheap to argue about, cheap to fix, with the fix itself
subject to the same public process.

## The practical layer

None of this is hypothetical, because the ground has already moved. Distributed consumer
compute networks — Akash, io.net, Render and projects in that family — rent idle GPU
capacity at fifty to eighty percent below hyperscaler rates, and open-weight models run
on them today. Put the two together and the need for a single gatekeeper weakens: a
model that can be downloaded, a machine that can be rented without permission, a
deployment path that needs nobody's approval.

The vision is a global inference and training network in which commodity hardware
contributes alongside datacenters — the workstation under a desk, the card in a
hobbyist's machine, the accelerator in a small lab — and in which models are designed to
run efficiently on hardware people actually own. That last clause is the engineering
part: every factor of compression, every megabyte saved, every byte of state that can be
streamed rather than held resident, is a household that can host instead of a household
that must rent. Efficiency is not a nicety here; it is the distribution mechanism.

## The posture is building

Nothing here is a project to shut anything down, and nothing here is a claim about the
people inside the large labs — most of them are doing careful, difficult work under real
constraints. The claim is about structure: a capability only one kind of institution can
hold will be held in ways its users did not choose, however good the intentions of the
people holding it.

So the aim is simple to state and hard to do: make the open path so capable, so cheap,
and so trustworthy that the closed path loses its reason to exist. People will choose the
system they can inspect, repair, host, and argue about — when it is also good. That is
why the technical work here is not separate from the civic question but part of the
answer to it: an engine that keeps a conversation unbounded on a single consumer card,
that streams experts from RAM and NVMe rather than demanding a rack, that measures its
own real capacity and refuses to report a number it cannot keep.

Democracy in AI is not a slogan. It is an architecture: a set of choices about who can
read the code, who pays for the research, who decides the rules, who owns the machines,
and how every one of those decisions is made visible. Design decisions can be made
differently. That is the whole bet.

# Which members should care management call next quarter

**For:** the director of care management at a Medicare health plan, deciding where a
fixed outreach team spends the next quarter.

**Question:** you can proactively contact a few thousand members. Which ones, and is it
worth doing?

## The situation

Spending is not spread evenly across the membership and never has been. In the year you
are planning against, the most expensive 10 percent of members accounted for 49.93
percent of everything the plan paid, and the most expensive 1 percent accounted for 13.47
percent. Half the budget sits with a tenth of the people.

That concentration is the whole opportunity. It is also the whole problem, because the
expensive members are only obvious in hindsight. By the time someone is in the top decile
of this year's spending, this year's money is gone. The question is who will be expensive
*next* year, while there is still time to intervene.

## What the team does today

Two things, both reasonable, both beatable.

The first is to work from last year's spending: pull the members who cost the most and
call them. The second is to work from the problem list: pull the members carrying the
most chronic conditions.

Both are real strategies and neither is stupid. Ranking by last year's cost reaches 19.92
percent of next year's spending in the top 10 percent of members. Ranking by chronic
condition count reaches 20.07 percent. Against random outreach at 10 percent, either is
roughly a doubling.

## What the model adds

The risk model reaches **23.97 percent** of next year's spending in the same top 10
percent, and finds 30.54 percent of the members who will actually land in next year's
high cost decile.

That is about 4 percentage points of captured spending above the better of the two
existing rules. On this membership, where total spending in the target year is $356.8
million, an outreach list of the same size sits in front of $85.5 million of next year's
spending instead of $71.6 million. Same team, same number of calls, $13.9 million more
spending in view.

Two honest qualifications belong right here.

The model is not close to perfect and cannot be. Perfect foresight would reach 49.93
percent in the top decile. The model closes about a third of the distance between random
and perfect. Next year's catastrophic cases are dominated by events that have not
happened yet, and no model built on claims history will see them coming.

The improvement also does not come from where you would expect. It does not come from the
chronic condition flags. It comes almost entirely from prior utilisation: how much the
member spent, how many professional claims they generated, how many prescriptions they
filled. Diagnosed conditions do separate members when you look at them one at a time,
but they add nothing once utilisation is in the model. In this dataset the problem list
is not carrying information that the claim history has not already given you.

## The tiers

The score is cut into three tiers so the team has something to act on.

| Tier | Members | Cost per member next year | Share of spend | Concentration |
|---|---|---|---|---|
| Low | 90,276 | $2,319 | 58.68% | 0.73x |
| Medium | 16,926 | $5,808 | 27.55% | 1.84x |
| High | 5,643 | $8,708 | 13.77% | 2.75x |

A member in the high tier costs 3.8 times what a low tier member costs. The high tier is
5 percent of the membership and holds nearly 14 percent of the spending.

## Is the programme worth running

This is where the analysis stops being able to answer the question on its own, and it is
important to be clear about why.

The claims data knows what members cost. It does not know what an outreach call costs,
how many members pick up the phone, or how much spending a care manager actually
prevents. Those three numbers decide the entire business case and none of them are in the
data. I assumed them:

- $500 to reach one member
- 35 percent of targeted members engage
- 10 percent of an engaged member's spending is avoidable

On those assumptions, calling the 5,643 members in the high tier reaches $49.1 million of
next year's spending, avoids $1.72 million of it, and costs $2.82 million to do. **It
loses about $1.10 million.** Calling 5,643 members at random reaches $17.8 million,
avoids $625,000, and loses $2.20 million.

So targeting is worth $1.10 million against random selection, and the programme still
does not pay for itself.

**The number to argue about is 16.41 percent.** That is the share of the high tier's
spending that proactive management would have to avoid for the programme to break even at
$500 per member and 35 percent engagement. I assumed 10 percent. If the team's own
experience says a well run programme avoids 16 percent or more of the spending of the
members it engages, this is worth funding. If it says 10 percent, it is not, at this
outreach cost.

That reframes the decision usefully. The question is no longer "is the model good", it is
"can you avoid a sixth of the spending of the people you reach, or can you get the cost per
contact down". Both of those are questions the care management team can answer from its
own programme data, and neither depends on the model at all.

## What I would ask for next

**The programme's own numbers.** Cost per completed contact, engagement rate by outreach
channel, and any historical before and after spending comparison on managed members. All
three assumed inputs above would become measured ones, and the business case would stop
being a sensitivity grid and start being an answer.

**Anything that is not claims history.** The model has hit the ceiling of what billing
records can tell it. Lab values, medication adherence, functional status, a discharge
feed, or anything with a shorter lag would add signal that prior spending does not
already contain.

**A shorter refresh cycle.** Scoring on a calendar year lag means the most recent
information about a member is up to twelve months stale. Rolling twelve month features
refreshed monthly would catch members whose utilisation turned upward recently, which is
exactly the group where intervention has the most room to work.

## The caveat that applies to all of it

This analysis is built on CMS synthetic claims. The file mirrors the structure of real
Medicare data closely enough that the pipeline would run unchanged against a real limited
data set, but CMS deliberately altered the statistical relationships between variables to
protect privacy. The concentration of spending, the two part structure and the evaluation
design all transfer. The specific finding that chronic conditions add nothing over
utilisation is most likely an artefact of that alteration, and I would expect a real
population to behave differently. It should be re-run on real data before anyone acts on
the tiering.

# Labelling rubric: "important enough to interrupt"

Write the rubric before annotating, pilot it, measure agreement, fix the rubric,
and only then annotate at scale. A rubric written after seeing the model's
scores is not a rubric.

## The question

**Would interrupting Rio right now be the right call?**

Not "is this email relevant", not "would Rio want to read this eventually".
Interruption is the action being labelled, so the label has to be about the
interruption.

## The persona

Labels are meaningless without it. `persona.py` in `evals/contamination/` is the
recipient: Rio Tanaka, engineering lead at a 40-person company, shipping a
checkout rewrite, hiring one backend engineer, renewing a cloud contract, the
only person who can approve a production deploy, approves contractor invoices
under $5,000. Read it before labelling and keep it open while you do.

This is why no public corpus can carry importance labels: importance is a
property of the (message, recipient, moment) triple, and a stranger cannot
annotate it from outside. Publishing the persona is what makes the label
reproducible.

## Label 1 (interrupt) if **any** of these hold

1. **A delay of a few hours causes harm.** A window closes, a deployment slips,
   a candidate takes another offer, money is lost.
2. **A decision only this person can make is being waited on.** Someone is
   blocked. Note that being blocked is about the sender's state, not the
   sender's tone.
3. **An account is being accessed.** One-time codes, sign-in alerts, password
   resets. These are labelled 1 even when the email is automated and even when
   the consequence of a false positive is another notification.

## Label 0 (do not interrupt) otherwise

Including, explicitly:

- Anything whose only claim on attention is urgent *wording*. "FINAL NOTICE",
  "ACT NOW", "expires tonight" are properties of the sender's marketing, not of
  Rio's day. If removing the urgent words would make it obviously unimportant,
  it is 0.
- Anything that needs a response eventually but not today.
- Anything informational: newsletters, digests, receipts, build notifications.

## Edge cases, decided in advance

These are the ones annotators will disagree on, so they are settled here rather
than case by case.

| Case | Label | Why |
| --- | --- | --- |
| A real deadline that is more than a week away | 0 | A few hours' delay causes no harm. Importance is not the same as consequence. |
| A quiet message with a same-day deadline buried in it | 1 | Tone is not the signal. Rule 1 does not mention wording. |
| An automated alert about a system Rio does not own | 0 | Nobody is blocked on Rio. |
| A security alert for an account that is not Rio's | 0 | Rule 3 is about *this person's* accounts. |
| A thread where the last message is "thanks" | 0 | Nothing is being waited on. |
| A meeting invite for next month | 0 | No harm in a few hours. |
| A meeting **change** for later today | 1 | The plan Rio already made is now wrong. |
| A request Rio has already answered in an earlier message | 0 | Nothing is waiting. |
| An invoice under $5,000 with a payment run closing today | 1 | Rule 1 and rule 2. |
| An invoice under $5,000 with net-30 terms | 0 | A few hours change nothing. |

## Procedure

1. **Label blind to the model's score.** `annotate.py` never shows it. If you
   have already seen a score for an email, hand that email to someone else.
2. **Double-label at least 50 items** and report Cohen's kappa before reporting
   any accuracy number. Your own label noise bounds every figure downstream. The
   only published email-importance corpus (ACL 2015, Enron, three levels)
   reached 0.64-0.80 by annotator tier; below about 0.7 the rubric is the
   problem, not the annotator, so fix the rubric and re-label.
3. **Stratify and over-sample the positive class.** At a 15% base rate, 200
   items gives about 30 positives and recall carries an interval of roughly
   ±11 points. Recall on the rare class is the binding constraint, and it is the
   error type that matters most here.
4. **Pre-register the cost ratio** — how many extra interruptions you would
   accept to catch one more important email — before looking at any score. It
   decides the threshold, and choosing it afterwards is choosing the answer.

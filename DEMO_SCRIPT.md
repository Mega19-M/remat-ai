# Demo video script (target 2:30, under 3:00)

Record with the **Live** mode on if possible. If you must use Demo mode, say so out loud on screen; do not present it as Decider output.

| Time | Screen | Say |
|---|---|---|
| 0:00 | Overview page | "Most waste apps just file a report. RE:MAT AI decides the right next investigation step, and refuses to guess when it is unsafe." |
| 0:20 | Architecture diagram | "A Strands Decider 2B model picks the next tool. A deterministic guardrail can veto it. Bedrock writes a labelled, unverified summary." |
| 0:40 | Scenarios → Agricultural residue → Run | "A routine case: the Decider picks recovery first. Findings carry source labels." |
| 1:10 | Decision trace | "Here are the options offered, the action the Decider selected, its confidence, and the tool that actually ran." (Only mention confidence if shown.) |
| 1:35 | Scenarios → Unknown chemical → Run | "Now an unlabelled drum. The user wants value, so a recovery step is proposed." |
| 1:55 | Trace, red guardrail box | "The safety rule rejects recovery and escalates. That override is shown as a safety override, not a Decider choice." |
| 2:15 | Battery emergency scenario | "Emergency wording sends us straight to escalation with a clear warning." |
| 2:30 | Recommendations | "No facility is ever invented. Everything is labelled source-backed, hypothesis or unknown. Thank you." |

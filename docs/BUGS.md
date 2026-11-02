# Three real bugs hit while building

## 1. After a failed booking, the agent stayed in "awaiting approval"

**What broke.** When `book` returned `slot_taken`, or `reschedule` returned `slot_conflict`, the agent correctly voided the approval and showed the remaining options. However, the workflow phase stayed `awaiting_approval`. The user's next message ("2") was then handled as a reply to a proposal that no longer existed, instead of as a choice of option.

**Why.** `_drop_option` rebuilt the option list but only changed the phase when the list became empty. The non-empty path assumed the caller was already in `choosing`, which is true when a hold fails but not when execution fails.

**Fix.** `_drop_option` now always moves to `choosing` when options remain (`errand_agent/agent.py`, `_drop_option`).

**Regression tests.** `test_slot_taken` asserts `phase == choosing` and that the taken slot is not offered again. `test_reschedule_slot_conflict` picks the next slot after a conflict and completes the reschedule.

## 2. The agent's own hold made it report a free slot as "just taken"

**What broke.** The user picks option 1, then says "no", then picks option 1 again. The second `hold_slot` failed with `slot_unavailable`, so the agent said "That slot was just taken" and dropped an option the user could book.

**Why.** The contract has no `release_hold`. The first hold is still active for 10 minutes, and the mock correctly treats a held slot as unavailable, including to the holder.

**Fix.** The workflow state remembers our live holds (`state.holds`). `_hold()` reuses an unexpired hold for the same provider and slot. Renewing after expiry forces a fresh hold, and a fresh hold re-checks the price.

**Regression test.** `test_repick_same_option_reuses_own_hold` asserts one `hold_slot` call, the same `hold_id` on both proposals, and a successful booking.

## 3. A forgotten area was still used after a restart

**What broke.** The user stated Indiranagar while booking a plumber, corrected their saved area to Koramangala, then said "forget my area". After a restart, they asked for a carpenter. The agent didn't ask for an area and carried on with Indiranagar, which was the area the user had told us to forget. The eval caught it because no `ask` event was emitted.

**Why.** Forgetting tombstoned the memory fact, but the area also lived in the persisted request in progress, marked as stated by the user in that request. Because the workflow resumes after a restart, the "forgotten" value survived there. The rule that forgotten facts are never reused was enforced in memory but not in the workflow.

**Fix.** `_forget` also clears the matching field from the request in progress, whatever its source. If options or a proposal were based on that field, it voids them.

**Regression test.** `test_memory_correct_and_forget` forgets the area, restarts, starts a new booking, and asserts that the agent asks for the area and the request has no area.

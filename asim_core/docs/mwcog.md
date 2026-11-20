# Second region: MWCOG

The two ported steps, `compute_accessibility` and `auto_ownership`, were run
on the MWCOG example (`activitysim/examples/prototype_mwcog`) with no new
modeling features. The goal: find out whether anything in asim_core was
specific to MTC.

## Summary

- **Both steps match 1.x exactly on MWCOG.** That includes all 29,972
  households, with every utility and probability bit for bit, and an end-to-end
  run with the other 35 steps replayed.
- **The test found one MTC-specific assumption in asim_core,** about how zone
  numbers map to skim rows. It's fixed, and asim_core now follows 1.x's rules.
  It also found two harness bugs, both fixed.
- **ActivitySim 1.x itself crashes on MWCOG with all households,** in
  `mandatory_tour_frequency`. That's a 1.x bug; details below.

## How MWCOG differs from MTC

| | MTC | MWCOG |
|---|---|---|
| Zones | 25 | 53 |
| Households in the example data | 5,000 | 29,972 (the example samples 100) |
| Skims | 826 matrices, `skims.omx` | 1,382 matrices, found by the pattern `skims*.omx` |
| Zone lookup in the OMX file | none | `zone_number`: 1, 2, 7, 8, ... (not consecutive) |
| Zone IDs in the tables | renumbered from 0 | original zone numbers |
| Time periods | EA, AM, MD, PM, EV | NT, AM, MD, PM (NT twice) |
| Accessibility | 22 skims; runs after households are loaded | 19 skims; runs before households are loaded |
| Auto ownership | 29 terms, county constants | 24 terms, different coefficients |

## Results

1.x was run twice. One run used all 29,972 households, with every step saved;
it stopped at `mandatory_tour_frequency` because of the 1.x bug below, after
saving 12 steps, including both ported ones. The other run used the
example's default 100-household sample and completed all 37 steps.

| Check | Compared | Result |
|---|---|---|
| `compute_accessibility`, from 1.x's saved state | 10 columns × 53 zones | exact |
| `auto_ownership`, all households | 29,972 households | exact |
| `auto_ownership` utilities and probabilities, all households | 149,860 each | bit-for-bit identical |
| Same checks on the 100-household run | 100 households, 53 zones | exact |
| End to end, 100 households: 2 steps computed, 35 replayed | after `cdap_simulate`, after `joint_tour_frequency`, at the end | exact |

Car ownership shares, all households (identical in both): 0 cars 47.73%,
1 car 27.94%, 2 cars 24.21%, 3 cars 0.09%, 4 cars 0.02%.

## What the test found

1. **Zone numbering was MTC-specific in asim_core.**
   - In 1.x, zones are renumbered from 0 only when *both* of these are set:
     `recode_columns: {zone_id: zero-based}` on the land-use table, and
     `recode_pipeline_columns: True`, which is off by default.
   - MTC sets both. MWCOG lists `recode_columns` but leaves
     `recode_pipeline_columns` off, so its tables keep the original zone
     numbers, and 1.x finds skim rows through the OMX file's zone lookup.
   - asim_core only checked `recode_columns`, so it treated MWCOG's zone
     numbers as matrix positions and stopped with "zone IDs outside 0..52".
   - Now it applies 1.x's rule. With renumbering, zone IDs are matrix
     positions, and a startup check confirms the OMX zone lookup lists
     land_use's original zone numbers in matrix order. Without renumbering,
     zone numbers are matched through the OMX lookup; files without a lookup
     use matrix positions, as 1.x does.
2. **Skim cache cleanup was too broad.** Two models sharing a cache folder
   would delete each other's caches. Cleanup now removes only older caches
   made from the same OMX files, and the harness keeps one cache folder per
   example, as 1.x keeps its cache under each run's output folder.
3. **The profiling harness ignored the example folder** and always profiled
   MTC. Fixed for both the milestone and experiment scripts.

## A 1.x bug

With all MWCOG households, 1.x stops in `mandatory_tour_frequency`:

- Three households (11310, 14815, 18648) end up with every probability
  zero.
- With `skip_failed_choices` on (1.x's default), 1.x marks their choices
  as -99 and reports them as skipped.
- The same step then converts choice numbers to alternative names with
  `model_spec.columns[choices.values]`
  (`activitysim/abm/models/mandatory_tour_frequency.py:136`). A -99 isn't a
  valid position, so the run stops with `IndexError: index -99 is out of
  bounds`.

The example's default 100-household sample never reaches these households.
asim_core stops at this point by design, naming the households, instead of
skipping them (see milestone1.md, "One deliberate difference"). This could
be reported to the ActivitySim project.

## Profiles

100-household run; each step in a fresh process from 1.x's saved state, 5
runs, medians. Skims prepared with MWCOG's own settings
(`read_skim_cache: False`): 1.x reads the 1,382 matrices from OMX and asim_core
imports them, every run. Sharrow was not run for MWCOG: its first compile
takes half an hour or more, and MWCOG's sharrow settings renumber zones
differently.

| Step | Engine | Step (s) | Extra memory (MB) | Skims ready (s) | Inside the full 1.x run (s) |
|---|---|---|---|---|---|
| compute_accessibility | asim_core | 0.004 | 1.0 | 0.56 | |
| compute_accessibility | 1.x legacy | 0.055 | 1.6 | 0.67 | 0.42 |
| auto_ownership | asim_core | 0.005 | 1.3 | 0.56 | |
| auto_ownership | 1.x legacy | 0.114 | 5.1 | 0.68 | 0.08 |

The full 1.x run of 100 households took 62 seconds, with a peak of 502 MB.

## Not covered by this test

MWCOG's auto ownership model is a simple multinomial logit, like MTC's, so
this test doesn't cover nested logit. ARC's auto ownership is the next test
for that. Preprocessors, two- and three-zone systems and the remaining steps
are also untested here.

## Reproducing

From the repository root:

```bash
python asim_core/harness/run_asim1.py --example-dir activitysim/examples/prototype_mwcog --out asim_core/work/mwcog/asim1_legacy
```

```bash
python asim_core/harness/run_milestone1.py --example-dir activitysim/examples/prototype_mwcog --asim1-out asim_core/work/mwcog/asim1_legacy --asim1-sharrow-out asim_core/work/mwcog/no_sharrow_run --out asim_core/work/mwcog/milestone1
```

For the all-households check, add `--households 0` to the first command
(1.x then stops at `mandatory_tour_frequency`, after both ported steps).

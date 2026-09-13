# Test results

Generated 2026-09-13 18:06 UTC by `python Test/run_test.py`.

**5 people enrolled** from 10 photographs, then 8 probes identified against them (5 genuine, 3 impostor). Engine `dlib`, threshold 0.6.

## Summary

| metric | value |
|---|---:|
| Accuracy | **100.00%** |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| False accept rate | 0.00% |
| False reject rate | 0.00% |

TP 5 · FP 0 · TN 3 · FN 0

## Enrolled

| person | photos |
|---|---:|
| Buzz Aldrin | 2 |
| Grace Hopper | 2 |
| Katherine Johnson | 2 |
| Mae Jemison | 2 |
| Sally Ride | 2 |

## Every probe

Probe photographs are different images from the enrolment ones.
`unknown_*` are people who were never enrolled and must be rejected.

| probe | expected | predicted | distance | runner-up | |
|---|---|---|---:|---|---|
| `Buzz_Aldrin_1.jpg` | Buzz Aldrin | Buzz Aldrin | 0.558 | Sally Ride (0.580) | PASS |
| `Grace_Hopper_1.jpg` | Grace Hopper | Grace Hopper | 0.526 | Katherine Johnson (0.599) | PASS |
| `Katherine_Johnson_1.jpg` | Katherine Johnson | Katherine Johnson | 0.337 | Grace Hopper (0.650) | PASS |
| `Mae_Jemison_1.jpg` | Mae Jemison | Mae Jemison | 0.356 | Katherine Johnson (0.702) | PASS |
| `Sally_Ride_1.jpg` | Sally Ride | Sally Ride | 0.073 | Katherine Johnson (0.694) | PASS |
| `unknown_1_1.jpg` | Unknown | Unknown | 0.748 | Sally Ride (0.784) | PASS |
| `unknown_2_1.jpg` | Unknown | Unknown | 0.631 | Sally Ride (0.652) | PASS |
| `unknown_3_1.jpg` | Unknown | Unknown | 0.670 | Sally Ride (0.719) | PASS |

The runner-up column is the next closest enrolled person. A large gap between the match and the runner-up means the decision was not marginal.

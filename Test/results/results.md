# Test results

Generated 2026-09-13 20:01 UTC by `python Test/run_test.py`.

**5 people enrolled** from 14 photographs, then 8 probes identified against them (5 genuine, 3 impostor). Engine `opencv`, threshold 1.012.

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
| Buzz Aldrin | 3 |
| Grace Hopper | 3 |
| Katherine Johnson | 3 |
| Mae Jemison | 3 |
| Sally Ride | 2 |

## Every probe

Probe photographs are different images from the enrolment ones.
`unknown_*` are people who were never enrolled and must be rejected.

| probe | expected | predicted | distance | runner-up | |
|---|---|---|---:|---|---|
| `Buzz_Aldrin_1.jpg` | Buzz Aldrin | Buzz Aldrin | 0.595 | Sally Ride (1.301) | PASS |
| `Grace_Hopper_1.jpg` | Grace Hopper | Grace Hopper | 0.696 | Katherine Johnson (1.266) | PASS |
| `Katherine_Johnson_1.jpg` | Katherine Johnson | Katherine Johnson | 0.836 | Buzz Aldrin (1.258) | PASS |
| `Mae_Jemison_1.jpg` | Mae Jemison | Mae Jemison | 0.842 | Grace Hopper (1.300) | PASS |
| `Sally_Ride_1.jpg` | Sally Ride | Sally Ride | 0.994 | Buzz Aldrin (1.320) | PASS |
| `unknown_1_1.jpg` | Unknown | Unknown | 1.277 | Grace Hopper (1.335) | PASS |
| `unknown_2_1.jpg` | Unknown | Unknown | 1.265 | Buzz Aldrin (1.315) | PASS |
| `unknown_3_1.jpg` | Unknown | Unknown | 1.242 | Sally Ride (1.255) | PASS |

The runner-up column is the next closest enrolled person. A large gap between the match and the runner-up means the decision was not marginal.

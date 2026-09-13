# Test results

Generated 2026-09-13 20:38 UTC by `python Test/run_test.py`.

**15 people enrolled** from 41 photographs, then 32 probes identified against them (22 genuine, 10 impostor). Engine `opencv`, threshold 1.012.

## Summary

| metric | value |
|---|---:|
| Accuracy | **87.50%** |
| 95% confidence interval | 71.9% – 95.0% |
| Precision | 1.0000 |
| Recall | 0.8182 |
| F1 | 0.9000 |
| False accept rate | 0.00% |
| False reject rate | 18.18% |

TP 18 · FP 0 · TN 10 · FN 4

## Enrolled

| person | photos |
|---|---:|
| Buzz Aldrin | 3 |
| Eileen Collins | 3 |
| Ellen Ochoa | 3 |
| Grace Hopper | 3 |
| Jessica Meir | 3 |
| John Glenn | 3 |
| Katherine Johnson | 3 |
| Mae Jemison | 3 |
| Michael Collins | 1 |
| Neil Armstrong | 3 |
| Peggy Whitson | 3 |
| Sally Ride | 2 |
| Scott Kelly | 3 |
| Sunita Williams | 2 |
| Victor Glover | 3 |

## Every probe

Probe photographs are different images from the enrolment ones.
`unknown_*` are people who were never enrolled and must be rejected.

| probe | expected | predicted | distance | runner-up | |
|---|---|---|---:|---|---|
| `Buzz_Aldrin_1.jpg` | Buzz Aldrin | Buzz Aldrin | 0.595 | Michael Collins (1.268) | PASS |
| `Ellen_Ochoa_1.jpg` | Ellen Ochoa | Ellen Ochoa | 0.683 | Eileen Collins (1.186) | PASS |
| `Grace_Hopper_1.jpg` | Grace Hopper | Grace Hopper | 0.696 | Neil Armstrong (1.249) | PASS |
| `Grace_Hopper_2.jpg` | Grace Hopper | Grace Hopper | 0.493 | Katherine Johnson (1.225) | PASS |
| `Jessica_Meir_1.jpg` | Jessica Meir | Jessica Meir | 0.491 | Peggy Whitson (1.187) | PASS |
| `Jessica_Meir_2.jpg` | Jessica Meir | Jessica Meir | 0.527 | Neil Armstrong (1.131) | PASS |
| `John_Glenn_1.jpg` | John Glenn | Unknown | 1.242 | Peggy Whitson (1.247) | FAIL |
| `John_Glenn_2.jpg` | John Glenn | Unknown | 1.063 | Neil Armstrong (1.176) | FAIL |
| `Katherine_Johnson_1.jpg` | Katherine Johnson | Katherine Johnson | 0.836 | Buzz Aldrin (1.258) | PASS |
| `Katherine_Johnson_2.jpg` | Katherine Johnson | Katherine Johnson | 0.674 | Grace Hopper (1.244) | PASS |
| `Mae_Jemison_1.jpg` | Mae Jemison | Mae Jemison | 0.842 | Grace Hopper (1.300) | PASS |
| `Mae_Jemison_2.jpg` | Mae Jemison | Mae Jemison | 0.834 | Sally Ride (1.251) | PASS |
| `Neil_Armstrong_1.jpg` | Neil Armstrong | Neil Armstrong | 0.722 | Peggy Whitson (1.250) | PASS |
| `Neil_Armstrong_2.jpg` | Neil Armstrong | Neil Armstrong | 0.556 | Victor Glover (1.231) | PASS |
| `Peggy_Whitson_1.jpg` | Peggy Whitson | Unknown | 1.023 | Ellen Ochoa (1.222) | FAIL |
| `Peggy_Whitson_2.jpg` | Peggy Whitson | Peggy Whitson | 0.760 | Sally Ride (1.260) | PASS |
| `Sally_Ride_1.jpg` | Sally Ride | Sally Ride | 0.994 | Peggy Whitson (1.186) | PASS |
| `Sally_Ride_2.jpg` | Sally Ride | Sally Ride | 0.934 | John Glenn (1.264) | PASS |
| `Scott_Kelly_1.jpg` | Scott Kelly | Scott Kelly | 0.660 | Neil Armstrong (1.236) | PASS |
| `Scott_Kelly_2.jpg` | Scott Kelly | Scott Kelly | 0.441 | Neil Armstrong (1.183) | PASS |
| `Victor_Glover_1.jpg` | Victor Glover | Victor Glover | 0.615 | John Glenn (1.252) | PASS |
| `Victor_Glover_2.jpg` | Victor Glover | Unknown | 1.059 | Scott Kelly (1.204) | FAIL |
| `unknown_10_1.jpg` | Unknown | Unknown | 1.218 | Sunita Williams (1.287) | PASS |
| `unknown_1_1.jpg` | Unknown | Unknown | 1.213 | Ellen Ochoa (1.256) | PASS |
| `unknown_2_1.jpg` | Unknown | Unknown | 1.226 | Katherine Johnson (1.265) | PASS |
| `unknown_3_1.jpg` | Unknown | Unknown | 1.089 | Scott Kelly (1.183) | PASS |
| `unknown_4_1.jpg` | Unknown | Unknown | 1.233 | Sally Ride (1.252) | PASS |
| `unknown_5_1.jpg` | Unknown | Unknown | 1.177 | Eileen Collins (1.213) | PASS |
| `unknown_6_1.jpg` | Unknown | Unknown | 1.179 | Sally Ride (1.226) | PASS |
| `unknown_7_1.jpg` | Unknown | Unknown | 1.237 | Neil Armstrong (1.262) | PASS |
| `unknown_8_1.jpg` | Unknown | Unknown | 1.124 | Peggy Whitson (1.241) | PASS |
| `unknown_9_1.jpg` | Unknown | Unknown | 1.189 | Mae Jemison (1.202) | PASS |

The runner-up column is the next closest enrolled person. A large gap between the match and the runner-up means the decision was not marginal.

## Failures

- `John_Glenn_1.jpg`: expected **John Glenn**, got **Unknown** at distance 1.242.
- `John_Glenn_2.jpg`: expected **John Glenn**, got **Unknown** at distance 1.063.
- `Peggy_Whitson_1.jpg`: expected **Peggy Whitson**, got **Unknown** at distance 1.023.
- `Victor_Glover_2.jpg`: expected **Victor Glover**, got **Unknown** at distance 1.059.

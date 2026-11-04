# distill-lab

What actually transfers when a small model learns from a bigger one?

I'm taking Qwen3-0.6B-Base as the student and Qwen3-4B as the teacher, and distilling grade-school math (GSM8K) into the student a few different ways:

- SFT on the human-written solutions, as a control
- SFT on the teacher's own solutions (response-based distillation)
- matching the teacher's token distributions with forward KL and with reverse KL (logit-based distillation)
- on-policy distillation, if I get that far

Every model gets the same evals. That covers accuracy on GSM8K and SVAMP, but also calibration, per-token agreement with the teacher, and a logit lens look at where in the network the answer forms. I'm more interested in *how* the students differ than in which one scores highest.

The losses and the training loop are written by hand in PyTorch. Everything runs on free Kaggle GPUs.

## baselines

before any training. 4-shot prompt, greedy decoding, up to 512 new tokens, one T4 on Kaggle.

| model | GSM8K (1,319) | SVAMP (1,000) |
|---|---|---|
| Qwen3-0.6B-Base, the student | 49.4% ± 2.7 | 63.0% ± 3.0 |
| Qwen3-4B, the teacher | 85.1% ± 1.9 | 84.0% ± 2.3 |

± is a 95% binomial interval. The gap on GSM8K is 36 points, which is the room distillation has to work with.

Every one of the 4,638 outputs had a number the extractor could read. 98% ended with the `#### answer` line from the prompt, the rest fell back to looking for "the answer is" or the last number. I read through the fallback cases by hand: almost all are answers that ran out of tokens mid-reasoning (the teacher likes to second-guess itself with "wait, ..."), which correctly count as wrong. I found one real misread, where the student wrote "deliver 160 red roses by 5 pm" and the extractor took the 5.

## teacher data

the teacher sampled 2 solutions (temperature 0.7) for each of the 7,465 training questions, same 4-shot prompt as above. I kept the ones with the right final answer.

| | |
|---|---|
| samples that were correct | 87.6% |
| questions with at least one correct solution | 6,904 (92.5%) |
| questions it never solved | 561 (7.5%) |
| solutions kept, after dropping 349 exact duplicates | 12,724 |
| kept solutions overlapping a test question | 0 |

so the response-distilled students never see the hardest 7.5% of problems. the human-solution student (S1) does.

then the teacher read its own kept solutions back and I saved its top 32 next-token log-probs at every solution token: 1,050,428 positions, 202 MB. all 151,936 log-probs would have been about 320 GB.

how much the top-32 cut loses, measured on 50 solutions with the teacher and the untrained student:

| k | teacher probability kept | forward KL error | reverse KL error | JSD error |
|---|---|---|---|---|
| 8 | 99.92% | -0.6% | -13.7% | -1.0% |
| 32 | 99.99% | -0.04% | -5.8% | -0.09% |
| 64 | ~100% | -0.02% | -4.1% | -0.04% |

the teacher is far more peaked than the 0.6B model was (top-32 kept 99.6% there), so for forward KL the cut is basically free. reverse KL is the odd one out. it's weighted by the student's probabilities, and an untrained student spreads mass over tokens the teacher gives almost nothing. the cache lumps all of those into one bucket, so it can't see exactly where that mass went and comes out about 6% low. it should shrink as the student gets closer to the teacher, which I'll check during training.

## SFT and response distillation

full fine-tune, 1 epoch, lr 1e-5 with warmup and cosine decay, batch of 32 sequences. trained students are evaluated zero-shot, since they've learned the answer format. S1 got a 4-shot run too, to make it directly comparable to the base model.

| model | trained on | prompt | GSM8K | SVAMP |
|---|---|---|---|---|
| base student | nothing | 4-shot | 49.4% ± 2.7 | 63.0% ± 3.0 |
| S1 | 7,469 human solutions | 0-shot | 52.2% ± 2.7 | 57.3% ± 3.1 |
| S1 | 7,469 human solutions | 4-shot | 48.3% ± 2.7 | 63.2% ± 3.0 |
| R1 | 12,724 teacher solutions | 0-shot | 63.5% ± 2.6 | 62.6% ± 3.0 |

training on the human solutions did basically nothing. every S1 number is within noise of the base model. the zero-shot SVAMP drop I saw first goes away with the 4-shot prompt, so that was about the prompt and not lost skill.

training on the teacher's solutions is a different story: +14 points on GSM8K. it did see more sequences (two solutions for most questions, and 561 fewer questions), so the data-size runs later will check how much of that is just volume. the human solutions are terse ("3*4=12"), the teacher writes out each step, and that style seems a lot easier for a 0.6B model to pick up. SVAMP doesn't move though. whatever R1 learned is tied to GSM8K-style problems, at least so far.

R1 trained for 398 steps in 27 minutes on one T4. training ce went from 0.30 to about 0.22, already low from the start because the teacher's text is predictable.

## logit distillation

same 12,724 teacher solutions as R1, same steps and lr. the difference is the target: instead of only the next token of the teacher's text, the student matches the teacher's top-32 distribution at every position. loss is 0.9 × KL + 0.1 × CE on the text.

| model | loss | GSM8K | SVAMP |
|---|---|---|---|
| R1 | CE on teacher text | 63.5% ± 2.6 | 62.6% ± 3.0 |
| L1 | forward KL | 64.7% ± 2.6 | 68.0% ± 2.9 |
| L2 | reverse KL | 64.7% ± 2.6 | 70.3% ± 2.8 |

on GSM8K the three are the same within noise. the teacher's text already carries most of what helps there.

SVAMP is where they split. R1 didn't move from the base model (63.0%), L1 gained 5 points and L2 gained 7. the L2 vs R1 gap is well outside both intervals. same questions, same solutions, so the only extra thing L1 and L2 got is the teacher's probabilities over the other tokens it was considering. that seems to be what carries over to a different kind of problem.

I expected reverse KL to do worse, since it's mode-seeking and my cache under-counts it by about 6%. it came out slightly ahead, though the L1 vs L2 gap (2.3 points) is inside the noise.

the first L1 run ran out of memory on the T4. every loss was doing a full 152k-vocab fp32 softmax at every padded position, three times per batch. only computing it on the ~82 graded tokens per example fixed it, and L1 now trains in about the same time as R1.

| | L1 | L2 |
|---|---|---|
| kd loss, first → last 10 steps | 0.36 → 0.11 | 0.72 → 0.18 |
| ce on the text at the end | 0.32 | 0.40 |
| training time on one T4 | 29 min | 29 min |

## status

baselines, teacher data, S1, R1, L1 and L2 done. next: the same runs on 1k and 3k questions, to see how each method scales with data.

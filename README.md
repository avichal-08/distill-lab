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

## status

baselines and teacher data done. next: the training loop.

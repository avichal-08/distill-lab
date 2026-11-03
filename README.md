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

## status

baselines done. next: generating the teacher's solutions and caching its probabilities.

# distill-lab

What actually transfers when a small model learns from a bigger one?

I'm taking Qwen3-0.6B-Base as the student and Qwen3-4B as the teacher, and distilling grade-school math (GSM8K) into the student a few different ways:

- SFT on the human-written solutions, as a control
- SFT on the teacher's own solutions (response-based distillation)
- matching the teacher's token distributions with forward KL and with reverse KL (logit-based distillation)
- on-policy distillation, if I get that far

Every model gets the same evals. That covers accuracy on GSM8K and SVAMP, but also calibration, per-token agreement with the teacher, and a logit lens look at where in the network the answer forms. I'm more interested in *how* the students differ than in which one scores highest.

The losses and the training loop are written by hand in PyTorch. Everything runs on free Kaggle GPUs.

## status

just getting started. results will land here as runs finish.

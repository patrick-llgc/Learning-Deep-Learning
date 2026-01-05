# [Stepwise Internalization: From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838)

_January 2026_

tl;dr: A curriculum learning experience of iteratively absorbing CoT into language model itself.

#### Overall impression
Stepwise Internalization is a method designed to achieve implicit chain-of-thought reasoning by gradually removing intermediate reasoning steps during training, first tokens first absorbed. 

This work inspired later more influential work such as [Coconut](coconut.md).

#### Key ideas
- The primary difference between implicit CoT and No CoT lies in the use of intermediate reasoning steps as supervision during training. 
- There is a trade off between num of CoT tokens remaining and the accuracy. Completely absorb into the model sometimes does not work. 

#### Technical details
- <!--Summary of technical details, such as important training details, or bugs of previous benchmarks.-->

#### Notes
- <!--Questions and notes on how to improve the current work-->


# [RHO-1: Not All Tokens Are What You Need](https://arxiv.org/abs/2404.07965)

_January 2026_

tl;dr: Selective Language Modeling (SLM) only backprop loss on valueable tokens. 

#### Overall impression
"Rho" denotes selective modeling of tokens with higher information “density (ρ).

Rho-1 is a great way to wash pretraining dataset.

The paper belong to the idea family that training on high quality tokens matter a lot. This is again quality > quantity. The results look very promising with much reduced training compute (matching DeepSeekMath with only 3% of the pretraining tokens). 

#### Key ideas
- SLM: General idea is to train a reference model on clean data, rate tokens from a larger corpus by their values, and train target model to backprop on high level tokens.
- Loss trajectory": tokens fall into four categories based on their loss trajectory—persistent high loss (H→H), increasing loss (L→H), decreasing loss (H→L), and consistent low loss (L→L).

#### Technical details
- In practice, token selection can be implemented by ranking the tokens in a batch according to their excess loss and using only the top k% of tokens for training. 
- Clean corpus 2B, total math CPT corpus 14B. 

#### Notes
- <!--Questions and notes on how to improve the current work-->


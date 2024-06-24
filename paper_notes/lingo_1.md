# [LINGO-1: Exploring Natural Language for Autonomous Driving](https://wayve.ai/thinking/lingo-natural-language-autonomous-driving/)

_June 2024_

tl;dr: Open-loop AD commentator with LLM.

#### Overall impression
Lingo-1's commentary was not integrated with the driving model, and remains an open loop system. Lingo-1 is enhanced by the relase of Lingo-1X, by extending VLM model to VLX by adding referential segmentation as X. This is enhanced further by successor [Lingo-2](lingo_2.md) which is a VLA model and finally achieving close-loop.

This is the first step torward a fully explanable E2E system. The language model can be coupled with the driving model, offering a nice interface to the E2E blackbox.

> A critical aspect of integrating the language and driving models is grounding between them. The two main factors affecting driving performance are the ability of the language model to accurately interpret scenes using various input modalities and the proficiency of the driving model in translating mid-level reasoning into effective low-level planning.

#### Key ideas
- Why language?
	- Accelerates training
	- Offers explanability of E2E one model
	- Offers controllability of E2E one model

#### Technical details
- Summary of technical details, such as important training details, or bugs of previous benchmarks.

#### Notes
- Questions and notes on how to improve/revise the current work


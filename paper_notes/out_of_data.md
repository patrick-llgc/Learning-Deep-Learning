# [Will we run out of data? An analysis of the limits of scaling datasets in Machine Learning](https://arxiv.org/abs/2211.04325)

_February 2023_

tl;dr: We will run out of high quality language data by 2026, and low-quality language data will run out by 2030-2050. Image data will run out much slower, perhaps by 2030-2060. We need to boost data efficiency in the long run to keep ML algorithm going.

#### Overall impression
The paper is full of interesting statistics for language datasets (web, papers, books, etc), and inspiring statistical modeling of growth trends. 

The current trend of ever-growing ML model may slow down significantly if data efficiency is not drastically improved or new data sources become available. 

> The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. -- Richard Sutton, [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html#:~:text=The%20biggest%20lesson%20that%20can,cost%20per%20unit%20of%20computation.)

#### Key ideas
- This report mainly focuses on unlabeled data, as unsupervised/self-supervised models has successfully created foundation models that can be fine-tuned for several downstream tasks using small amount of labeled data and large amount of unlabeled data. 
- Possible future directions for scaling ML models
	- data efficiency. Now much less efficient than human being.
	- data sources. 
		- Simulation data and bridging sim2real gap
		- Major economic shift may impact the production of data, such as autonomous driving for images.
- About half of the progress made by LLM over the past 4 years can be contributed to data. --> Coming up with efficient data consuming architecture without significant bottleneck is the correct way to scale up ML models.


#### Technical details
- High quality language data sources has a common property that they pass usefulness/quality filters (QA). 
	- user generated contents by dedicated internet contributors, such as wikipedia. 
	- produced and peer reviewed by subject matter experts
- Optimal dataset size is **proportional to the square root** of the compute budget. This is for language model though. 
- Language training dataset doubles every 16 months, while vision is slower at 42 months (3.5 years). The largest language dataset has 3 orders of magnitude more data points than vision. 

![](https://cdn-images-1.medium.com/max/1600/1*qPY4zgLyIlRjw9CA6nMjrg.png)

#### Notes
- [blog post from Epoch.AI](https://epochai.org/blog/will-we-run-out-of-ml-data-evidence-from-projecting-dataset)

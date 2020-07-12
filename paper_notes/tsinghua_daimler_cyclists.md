# [TDC (Tsinghua-Daimler Cyclists): A New Benchmark for Vison-Based Cyclist Detection](http://www.gavrila.net/Publications/iv16_cyclist_benchmark.pdf)

_July 2020_

tl;dr: Cyclist dataset collected in Beijing (朝阳区+海淀区).

#### Overall impression
Related to [Specialized Cyclists](specialized_cyclists.md).

KITTI only has 1400 cyclists in the entire datasets.

Why cyclists are important?

- In some small and mid cities in China where cyclists appear often, even more road accidents involve cyclist. 
- Different appearances and sizes at different viewpoints. 
- They can move 25 mph, 5 times faster than a pedestrian (pedestrian [preferred moving speed](https://en.wikipedia.org/wiki/Preferred_walking_speed) is 1.4 m/s, or 3 mph)

#### Key ideas
- 30k images, 22k cyclist instances.
- In many pedestrian datasets, cyclists are ignored as they look like pedestrians. 
- Difficulty levels:
	- Easy: height > 60 pixel, fully visible
	- Medium: height > 45 pixel, less than 40% occluded
	- Hard: height > 30 pixel, and less than 80% occluded
- One bbox for the bicycle and person as a whole, and one bbox for the person alone. --> maybe this is an overkill?

#### Technical details
- Annotate orientation
![](https://cdn-images-1.medium.com/max/1600/1*6CNzPtVbFnJbamR-_0pCeA.png)
![](https://cdn-images-1.medium.com/max/1600/1*i3tugooDjnFn-LBqbk1Q4A.png)
![](https://cdn-images-1.medium.com/max/1600/1*a1uCQUhpJYQ5RemMbEsllg.png)

#### Notes
- Questions and notes on how to improve/revise the current work  


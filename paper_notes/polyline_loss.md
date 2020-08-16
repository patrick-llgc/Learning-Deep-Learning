# [Polyline Loss: Hierarchical Recurrent Attention Networks for Structured Online Maps](https://openaccess.thecvf.com/content_cvpr_2018/papers/Homayounfar_Hierarchical_Recurrent_Attention_CVPR_2018_paper.pdf)

_August 2020_

tl;dr: Proposed the idea of polyline loss to encourage neural network to output structured polylines. 

#### Overall impression
This is one application of RNN in boundary extraction. Previous work include [Polygon-RNN](http://www.cs.toronto.edu/polyrnn/poly_cvpr17/), [Polygon-RNN++](http://www.cs.toronto.edu/polyrnn/), [Curve GCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ling_Fast_Interactive_Object_Annotation_With_Curve-GCN_CVPR_2019_paper.pdf) also from Uber ATG. The main idea is to create a structured boundary to boost the efficiency for human-in-the-loop  annotation.

One key difference from [Polygon RNN](http://www.cs.toronto.edu/polyrnn/poly_cvpr17/) is that polygon RCNN uses the cross entropy loss to learn the position fo the vertices. This is not ideal as there is no unique way to draw the same polygon.

[Polyline Loss](polyline_loss.md) focuses on easier lane topology on highways, and [DAGMapper](dagmapper.md) focuses on highway driving, and focuses on hard cases like forks and merges. [Polymapper](polymapper.md) only focuses on extracting road network and do not have lane-level information. 

#### Key ideas
- Polyline loss: measures the deviation of the edges of the ground truth polylines and their predictions. This is more suitable than distance on vertices, as there exists many ways to draw equivalent polylines. 
![](https://cdn-images-1.medium.com/max/1600/1*NeGg78_hIZfQom5eqUMwmg.png)
- Why polylines?
	- In HD Maps, road lanes are structured objects and are typically represented as a set of polylines, one per lane boundary. 
	- Most motion planners can only handle lane graphs that re structured and represent the right topology.
- Two recurrent networks
	- One focuses on recurrently extracts the starting location of laneline location. This is recurrent attention for lane counting. 
	- The second one starts from the starting region (instead of an accurate point) and draws the polyline, one vertex after another. 
- Baseline: Dilate GT by 20 pixels wide and do semantic segmentation. This may have holes and discontinuity in them.

#### Technical details
- Resolution: 5 cm per pixel. Evaluation of up to 20 cm accuracy (4 pixels).
- As of 2018, the estimate to HD Map the entire united states is 2 Billion US dollars. 
- 0.91 recall within 20 cm accuracy.

#### Notes
- Questions and notes on how to improve/revise the current work  


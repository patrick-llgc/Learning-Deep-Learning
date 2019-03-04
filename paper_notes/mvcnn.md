# [Multi-view Convolutional Neural Networks for 3D Shape Recognition](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.pdf) (MVCNN)

_Mar 2019_

tl;dr: Aggregate CNN features from multiple 2D projections of a 3D object to obtain a high quality 3D feature. 

#### Overall impression
The idea of using pre-trained and fine-tuned CNN to extract 2D features has been widely used. This paper explores ways to effectively aggregate these 2D features (concatenation, average or max pooling) into a high quality feature for the 3D object. Given that this feature should be insensitive to the number of 2D projections and the permutation of the list of 2D features (orderless list), max pooling of 2D features across views seems a very natural choice. In addition, the learning of a low-rank Mahalanobis metric significantly boosts the retrieval performance.

#### Key ideas
- Humans learn from 2D view of 3D object.
- 3D is worse than 2D (for now)
	- GPU memory (3D has to be downsampled compared to 2D)
	- Pre-trained model available for 2D CNNs
	- 3D features are too high dimensional
	- 2D is robust to 3D shape representation artifacts (holes, imperfect mesh tessellation, etc)
- Simply concatenating or averaging features leads to inferior performance. **Max pooling** across views proved to be the best.
- Multiview representation (CNN) vs aggregated representation (MVCNN)
	- **Classification**: CNN method used sum of classification values from different views for final classification results. MVCNN has only one classification score from the aggregated features.
	- **Retrieval**: CNN method used minimum distance among $n_x \cdot n_y$ pairs. MVCNN could use the aggregated features for classification, but learning a low-rank metric to further increases performance in retrieval. MVCNN is more computationally efficient.
- The features are fine-tuned for classification and thus sub-optimal for retrieval task. A low-rank Mahalanobis metric is learned to basically reduce the dimension of features to a fixed 128 dimensional feature. 

#### Technical details
- Multiview inputs: either 12 views around z-axis (assuming all 3D objects are upright) or 20 views on a icosahedron (20 faces, 12 vertices) in 4 pi stereo space.


#### Notes
- **Mahalanobis distance** provides a way to measure how similar some set of conditions is to a known set of conditions. For example, given the overall statistics of height and weight in a population, and give two sample data of height and weight pair, calculate the distance of the two samples. 
	- Mahalanobis distance is dimensionless, and accounts for the correlation between variables. 马氏距离有很多优点，马氏距离不受量纲的影响，两点之间的马氏距离与原始数据的测量单位无关；由标准化数据和中心化数据(即原始数据与均值之差）计算出的二点之间的马氏距离相同。马氏距离还可以排除变量之间的相关性的干扰。 (from [CSDN](https://blog.csdn.net/u010167269/article/details/51627338))
 	- MD can be used to find outliers in a distribution.
	- Here is an example from [landscape analysis](http://www.jennessent.com/arcview/mahalanobis_description.htm)
	- Here is a nice illustration on what is MD is doing (rotating and scaling) from [stackexchange](https://stats.stackexchange.com/questions/62092/bottom-to-top-explanation-of-the-mahalanobis-distance). Essentially Mahalanobis distance is Euclidean distance in rotated and scaled space.
- [Mahalanobis metric leaning](http://www.uta.fi/sis/mtt/mtts1-dimensionality_reduction/drv_lecture9.pdf)
	
	> Thus learning Mahalanobis metric corresponds to learning a linear data transformation! If some eigenvalues (diagonals of D) are zero, then the metric effectively performs dimensionality reduction
	- It is quite similar to NCA (neighborhood component analysis) and different from PCA (unsupervised, find a direction on which the variation of the whole data set is the largest) and LCA (linear discriminant analysis, supervised, find a direction where the data from the same classes are clustered while from different classes are separated).
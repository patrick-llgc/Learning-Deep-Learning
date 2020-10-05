# [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

tl;dr: Focal loss solves the class imbalance problem by modifying the model with a new loss function that focuses on hard negative samples. Concretely, it modulates cross entropy loss by a L2 loss. 

- Focal loss can be used for classification, as shown [here](https://shaoanlu.wordpress.com/2017/08/16/applying-focal-loss-on-cats-vs-dogs-classification-task/). 

#### Takeaways

- Imbalanced training, balanced test: When trained on imblanced data (up to 100:1), the model trained with focal loss has evenly distributed prediction error when test data is balanced. 

- Imbalanced training, imbalanced test: traning with focal loss yields better accuracy than trained with cross entropy. Again it has evenly distributed prediction error. 
# [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

- tl;dr: Focal loss solves the class imlance problem by modifying the model with a new loss funciton that focuses on hard negative samples. 


Focal loss can be used for classification, as shown [here](https://shaoanlu.wordpress.com/2017/08/16/applying-focal-loss-on-cats-vs-dogs-classification-task/). The takeaway is:

- Imbalanced training, balanced test: When trained on imblanced data (up to 100:1), the model trained with focal loss has evenly distributed prediction error when test data is balanced. 

- Imbalanced training, imbalanced test: traning with focal loss yields better accuracy than trained with cross entropy. Again it has evenly distributed prediction error. 
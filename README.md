# CNN Using Pytorch (GPU or CPU)
This is an example of Convlolutional Neural Network using Pytorch (GPU or CPU). The data used here is MNIST from torchvision.datasets. The code split the data to training and validation data sets, then load it to training and validation loaders. The user has the choice to use the subclass and sequential model. The code:
1) generate loss and accuracy versus epoch and plot both in one graph.
2) plot the output for classified and misclassified validation data.
3) plot the Confusion matrix for the validation data.
The figure below show the loss and accuracy versus epoch for validation process: 

![loss_accuracy_epoch](https://user-images.githubusercontent.com/12114448/222929621-8bd00032-29e3-4d26-b5fd-4c4bb2d9d75b.png)

The output for classified and misclassified data (1st 100 sample):

![test_classified_validation](https://user-images.githubusercontent.com/12114448/222929628-659acb69-df30-4a1e-828c-31d84fbfdd2b.png)
![test_miclassified_validation](https://user-images.githubusercontent.com/12114448/222929630-b7ae32ff-ad2c-4095-a60c-a1d245a74a0a.png)

The confusion matrix for validattion data:

![confusion_matrix_validation](https://user-images.githubusercontent.com/12114448/222929639-f93a5e72-42a1-43b8-b15f-ea1e04a26d4e.png)

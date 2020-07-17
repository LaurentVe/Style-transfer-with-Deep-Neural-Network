# Style-transfer-with-Deep-Neural-Network
Use pre-trained VGG19 feature extractor section to apply style transfer to given image

## Description
In this notebook, i recreate style transfer method described in the paper "Image Style Transfer Using CNN" by Gatys.

The original paper can be found here: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

Style transfer relies on separating the content and style of an image. Given one content image and one style image, we aim to create a new, target image which should contain our desired content and style components:
- objects and their arrangement are similar to that of the content image
- style, colors, and textures are similar to that of the style image

## Dependencies
The model uses the VGG19 pretrained model, and more specifically its feature extractor section (all convolutional and pooling layers) while the classification section (the fully connceter layers) are not relevant for this task.

The VGG19 model is comprised of a series of convolutional and pooling layers, and a few fully-connected layers. The convolutional layers are grouped into five stacks, each stack being followed by a maxpooling layer. Each conv layer is named by stack and their order in the stack. Conv_1_1 is the first convolutional layer that an image is passed through, in the first stack. Conv_2_1 is the first convolutional layer in the second stack. The deepest convolutional layer in the network is conv_5_4.

A representation is shown below.

![](/notebook_ims/vgg19_convlayers.png)

## Process
Two input images are used, one for the style, one for the content. A copy of the content image is successively modified through iterations to incorporate style from the style image while maintaining the original content.

The feature extractor is used to extract style artifacts and content of given images. To get the content and style representations of an image, we pass an image forward through the VGG19 network until we get to the desired layers and then get the output from that layer.

Different layers through the model are used to provide the style artifacts and the content.
- Content: output of the Conv 4.2, ie the convolutional layer N°2 in the stack N°4.
- Style: weighted ouputs of the first conv layer in each of the 5 stacks.

Ponderation weigths can be modified in the notebook.

The model calculates two losses while adjusting the target image to the originals:
- Content loss: `content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)`
- Style loss: sum over layers of `style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)`

total loss : `total_loss = content_loss * content_weight + style_loss * style_weight`

More explanations are provided in the notebook.

## Content
- Load pre-trained VGG19
- Load Content and Style images
- Get content and style features from both images
- Compute Gram matrices
- Set content and style ponderation
- Update target image calculating the losses
- Display and save target image


## Installation
- download the repository
- insert your own style and content images in the "images" folder
- adjust the names of the images in the following line of codes:

`content = load_image('images/tour_eiffel.jpg').to(device)`

`style = load_image('images/delaunay.jpg', shape=content.shape[-2:]).to(device)`

- Give a name to the target image in this line of code. It will be saved in the working folder:

`im.save('eiffel_pic2.jpeg')`

Note:
- the style image is forced to be of the same shape as the content image: `shape=content.shape[-2:]`
- the images are resized to be of max size of 400 pixels

## Results

![](/notebook_ims/eiffel2_pic.jpeg)

![](/notebook_ims/essec_pic.jpeg)

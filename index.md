**Codebase:** [https://github.com/msbutler/cs205final](https://github.com/msbutler/cs205final)

# Problem Statement
Climate change has brought about increasingly frequent natural disasters, including flooding, thereby threatening human lives and infrastructure amongst others. Access to accurate and identifiable visual data is extremely instrumental for a directed and efficient relief response. The proliferation of unmanned aerial systems (UAS) with inexpensive sensors has led to a host of high resolution images but the main challenge of analysing these images on a high-frequency and real-time basis with high accuracy still stands. 

Therefore, our project aims to construct and train a deep image classifier system that identifies flooded regions from these UAS images. This project requires big compute due to protracted training time required for complex deep Convolutional Neural Networks (CNNs), as well as the need for more iterations without using a pre-trained model. Big data techniques are also employed to more effectively manage numerous large high-definition images. Hence, we deploy the model for fast, highly parallelizable classification on Amazon Web Services (AWS).


# Existing Work
Several studies in recent years have explored image classification for disaster relief. Gebrehiwot et al. (2019) developed a deep CNN for flood mapping based on UAV data (similar to our dataset).[[1]](#1) Gebrehiwot reported that it took about 26 hours for cross validation using a single GPU (NVIDIA Quadro M4000). Sarker et al. (2019) developed a supervised CNN for flood mapping based on satellite images.[[2]](#2) They used a High Performance Computing (HPC) server to train their model rather than a GPU and reported a training time of 24 hours. A more recent study, Hashemi-Beni and Gebrehiwot (2021), also developed a CNN for flood labeling of images, but did not report the infrastructure used for training the model or the required training time.[[3]](#3)

Instead of a fully-supervised approach as with the studies described above, we focused on building a semi-supervised model because real-world datasets are predominantly unlabelled. Our semi-supervised model is a convolutional neural network (CNN) that learns from both labeled and unlabeled images, therefore requiring a custom loss function and various data augmentations for implementation. Instead of transferring and applying a pre-trained model, we built our CNN from scratch, adapting and simplifying key components from Google's MixMatch and FixMatch algorithms.[[4]](#4)[[5]](#5) We elaborate on the specifics of our algorithm in the following section.

Additionally, it is worth noting that this work can be easily extended to other natural disasters like wildfires and debris from earthquakes. We hope this work contributes to the increasingly numerous humanitarian applications of machine learning with big compute and big data.


# Model and Data
## Model
We developed a custom semi-supervised CNN for image classification. The architecture used consists of [FILL IN WHEN FINALIZED] and is illustrated in **Figure 1** below.

**Figure 1: CNN Architecture**
[INSERT FIGURE 1: CNN Architecture (use AM231 PPT slide)]

During training, we use the architecture described above to classify both labeled and unlabeled images as flooded (1) or nonflooded (0). For labeled images, we want the model to produce predictions that match the ground-truth labels, so we penalize the loss function when predictions do not match the true labels. With unlabeled images, we want the model to assign photos with similar features to the same class. Therefore, we augment each unlabeled image several times and penalize the model for producing different predictions across augmentations of the same image. Augmentations may include rotations, translations, reflections, or noise additions to the original photos. For consistency between the labeled and unlabeled training data, a single augmented version of each labeled photo is used in lieu of the original image.  

The training process for a single mini-batch within an epoch is illustrated in **Figure 2**. We perform a forward pass on the augmented version of each labeled photo and the _k_ augmentations of each unlabeled photo within the batch. For the unlabeled images, a "guess" at the true label is produced by taking the mean of the model predictions for the _k_ augmentations of the same image. Model predictions are then evaluated using cross entropy with the true label (for the labeled data) or the L_2 loss with the guess label (for the unlabeled data). Total loss for the batch is the sum of the labeled loss with the unlabeled loss, where the unlabeled loss is weighted by a constant.

For training, we use the Adam optimizer with a learning rate of 0.001 and XXX epochs.

[FILL IN NUMBER OF EPOCHS in last sentence]

**Figure 2: Outline of Model Training for One Batch/Epoch**
![](figs/fig2.png)


## Data
The dataset used comes from the Floodnet Challenge [[6]](#6), with approximately 2,300 quadcopter or drone images of land from post-Hurricane Harvey. The data is segmented into 60% training, 20% validation and 20% testing sets. Of the training set, 25% is labeled (approximately 400 out of 1,400 images). Examples of a non-flooded and flooded image are shown in **Figure 3**. These images are of high resolution, 3000 by 4000 pixels, and hence are reduced to 1000 by 750 pixels for more efficient training and memory management.

**Figure 3: Example Images for Classification**
![](figs/fig3.png)

For training, we subset the images to create a balanced set of [FILL IN SPECIFICS ABOUT # OF IMAGES, BALANCED/UNBALANCED, ETC].


# Parallel Application, Programming Models, Platform and Infrastructure
Training convolutional neural networks is highly computationally intensive due to the many intermediate calculations required at each point in the architecture. In our situation, this issue is exacerbated by the high quality resolution of our images, which inherently increases the problem size at every intermediate step. Fortunately, matrix multiplication, convolutions, and pooling are all highly parallelizable tasks, and for this reason we relied on accelerated computing with a GPU to speed up the training and evaluation process for our model. This constitutes procedure-level parallelization as we are parallelizing regions of code within a task and thus falls in the external, fine-grained domain of Big Compute.

We evaluate performance by training our model several times using increasingly powerful instances of a single GPU on AWS. All configurations relied on Ubuntu 18.04 with the AWS Deep Learning AMI. **Table 1** includes a list of each configuration with additional details. We relied on Python (specifically Tensorflow) to build our CNN. Additionally, images are stored in an S3 bucket, also on AWS. 

[INSERT TABLE 1: List of GPU instances used - maybe add configuration details]


# Software Design
Technical description of the software design, code baseline, dependencies, how to use the code, and system and environment needed to reproduce tests

As mentioned above, our model is built in Python primarily using `tensorflow`. We also rely on the `os` package for reading in data; `skimage`, `random`, and `PIL` for image analysis; `numpy` for additional data analysis; and `matplotlib` for plotting our results. Each of these packages comes pre-installed with the AWS Deep Learning AMI. Replication information for producing the same environment and package versions used in our tests is included in the `Replication.md` instructions file on the Github (see Codebase link above).   

Our code structure is as follows:
- file1: description
- file2: description

Additionally, we rely on Tensorboard for GPU analysis...
[HOW TO USE TENSORBOARD - LINK TO TUTORIAL AS PART OF DESCRIPTION?]


# Performance Evaluation
Performance evaluation (speed-up, throughput, weak and strong scaling) and discussion about overheads and optimizations done


# Advanced Features
Description of advanced features like models/platforms not explained in class, advanced functions of modules, techniques to mitigate overheads, challenging parallelization or implementation aspects...

[TENSORBOARD USE]


# Discussion
goals achieved, improvements suggested, lessons learnt, future work, interesting insights


# Citations
<a id="1">[1]</a> 
Asmamaw Gebrehiwot et al. "Deep convolutional neural network for flood extent mappingusing unmanned aerial vehicles data". In: _Sensors_ 19.7 (2019), p. 1486.

<a id="2">[2]</a> 
Chandrama Sarker et al. "Flood Mapping with Convolutional Neural Networks Using Spatio-Contextual Pixel Information". In: _Remote Sens._ 11, 2331 (2019).

<a id="3">[3]</a> 
Leila Hashemi-Beni & Asmamaw Gebrehiwot. "Flood Extent Mapping: An Integrated Method Using Deep Learning and Region Growing Using UAV Optical Data". In: _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_ 14 (2021), p. 2127-2135.

<a id="4">[4]</a> 
David Berthelot et al. "Mixmatch: A holistic approach to semi-supervised learning". In: _arXivpreprint arXiv:1905.02249_ (2019).

<a id="5">[5]</a> 
Kihyuk Sohn et al. "Fixmatch: Simplifying semi-supervised learning with consistency andconfidence". In: _arXiv preprint arXiv:2001.07685_ (2020).

<a id="6">[6]</a> 
_IEEE Earth Vision 2021 Floodnet Challenge_. [http://www.classic.grss- ieee.org/earthvision2021/challenge.html](http://www.classic.grss- ieee.org/earthvision2021/challenge.html). Accessed: 2021-04-01.




Template Stuff:

## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/msbutler/cs205final.github.io/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Overview
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/msbutler/cs205final.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

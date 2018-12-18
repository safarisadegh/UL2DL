# Deep UL2DL
## Source code of:
Deep UL2DL: Channel Knowledge Tranfer from Uplink to Downlink

## Abstract

Knowledge of the channel state information (CSI) at the transmitter side is one of the primary sources of information that can be used for efficient allocation of wireless resources. Obtaining Down-Link (DL) CSI in FDD systems from Up-Link (UL) CSI is not as straightforward as TDD systems, and so usually users feedback the DL-CSI to the transmitter. To remove the need for feedback (and thus having less signaling overhead), several methods have been studied to estimate DL-CSI from UL-CSI. In this paper, we propose a scheme to infer DL-CSI by observing UL-CSI in which we use two recent deep neural network structures: a) Convolutional Neural network and b) Generative Adversarial Networks. The proposed deep network structures are first learning a latent model of the environment from the training data. Then, the resulted latent model is used to predict the DL-CSI from the UL-CSI. We have simulated the proposed scheme and evaluated its performance in a few network settings. 

![UL to DL knowledge transfer procedure](Images/model.png?raw=true "UL to DL knowledge transfer procedure")


## Direct Approach

![Direct approach:](Images/direct.png?raw=true "Direct approach:")

## Generative Approach

![Generative Approach Training](Images/ganstructure.png?raw=true "Generative Approach Training")


![Generative Approach Image Completion](Images/completion.png?raw=true "Generative Approach Image Completion")

## Datasets
To access datasets contact us: msadeq.safari@gmail.com

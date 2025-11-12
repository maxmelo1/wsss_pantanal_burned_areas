# Weakly Supervised Semantic Segmentation Burned Area Mapping

<h2 style="text-align: center;">LAGIRS 2025</h2>

The oficial repository of Puzzle-CAM fork for Burned Area Mapping.


## Abstract

Wildfire mapping in remote and ecologically sensitive regions like the Brazilian Pantanal faces challenges due to the high cost of collecting pixel-level annotations required for fully supervised models. In this study, we investigate the use of Weakly Supervised Semantic Segmentation (WSSS) methods—specifically SEAM and Puzzle-CAM—for burned area mapping using multispectral (RGB-NIR) satellite imagery. Both models were adapted to handle four-band data to leverage spectral information relevant for fire detection. Our two-stage pipeline first generates pseudo-labels from image-level annotations and then trains a SegFormer segmentation model on these labels. Experimental results show that Puzzle-CAM, particularly when combined with a stronger ResNeSt-101 backbone, produces high-quality pseudo-labels, leading to segmentation results that closely approach those of fully supervised methods. This approach demonstrates the potential of combining weak supervision and advanced network architectures to reduce labeling costs while enabling scalable wildfire monitoring across the Pantanal. Future work will focus on improving model robustness and extending the methodology to other types of ecological disturbances.

## Prerequisite

- Same configurattion and prerequisites of the oficial repository.
- 1x RTX with at least 12 GB's of VRAM.

## Usage

- Please, follow the instalation and instructions of the original [Puzzle-CAM repository](https://github.com/shjo-april/PuzzleCAM/). 



## Trained weights

- All model weights can be downloaded [here](https://drive.google.com/drive/folders/1pMyvV-3H0gLGBTa951rbAFB4FriKCtFZ?usp=sharing).


## TODO

- [ ] Add source code
- [ ] Add the dataset patches

## Contact

Maximilian Melo: [e-mail](mailto:maximilian.melo@ifms.edu.br).
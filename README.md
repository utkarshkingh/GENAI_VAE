# Image Compression and Generation using Variational Autoencoders

This project demonstrates image compression and generation using Variational Autoencoders (VAEs) implemented in Python with PyTorch. The main goal is to compress images into a latent space and then generate new images or reconstruct existing ones. The project uses a dataset of computer-generated font images from the [Character Font Images Data Set](https://archive.ics.uci.edu/ml/datasets/Character+Font+Images) available on the UCI Machine Learning Repository.

## Project Overview

- **Exploratory Data Analysis & Preprocessing:**  
  The project starts by exploring the dataset and preprocessing the images. Steps include converting images to grayscale, resizing to a uniform size, and transforming them into tensors.

- **Training/Validation Split & Data Loaders:**  
  The dataset is split into training and validation sets. Custom data loaders are then created to efficiently feed data into the model during training.

- **VAE Architecture:**  
  A Variational Autoencoder is implemented with fully connected layers. The encoder compresses the flattened images into a lower-dimensional latent space, and the decoder reconstructs the images from this latent representation. The model is trained using a combined loss function comprising binary cross-entropy (BCE) and Kullback-Leibler divergence (KLD).

- **Training Loop & Evaluation:**  
  The training loop includes forward and backward passes over multiple epochs. Reconstruction images and random samples from the latent space are periodically saved. Both training and validation losses are tracked and plotted.

- **Results & Analysis:**  
  The project outputs reconstructed images, newly generated samples from the latent space, and plots of the training and validation loss curves, which help in assessing the model's performance.

## Getting Started

### Prerequisites

- Python 3.11+
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://tqdm.github.io/)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/utkarshkingh/GENAI_VAE.git
   



2. Install the Required Packages
  You can install the required packages using pip:
   ```bash

   pip install torch torchvision matplotlib tqdm


3 . Dataset Preparation
    Download the Character Font Images Data Set from the UCI Machine Learning Repository or use your own collection of font images. Organize the images into a   
    folder structure similar to:

```console
Font/
  all/
    class1/
      image1.png
      image2.png
      ...
    class2/
      ...
```



# **Usage**

You can run the project in a Jupyter Notebook or as a standalone Python script. The notebook is structured into several tasks:

1.Data Exploration & Preprocessing:
  Visualize and preprocess images from the dataset.

2.Dataset Splitting & Data Loaders:
  Create a training/validation split and define data loaders.

3.VAE Model Definition:
  Build the VAE architecture, define the loss function, and instantiate the model.

4.Training Loop:
  Train the VAE while monitoring training and validation losses. The code also saves reconstruction images and random samples from the latent space.

5.Results Analysis:
  Visualize loss curves and inspect the output images to evaluate model performance.

To start training, simply run the notebook cells in order. You can adjust the number of epochs and hyperparameters as needed.


# **Project Structure**


```console

├── Font/                # Directory containing font images (organized by class)
├── Models/              # Folder to save model checkpoints and loss logs
├── Results/             # Folder to save generated and reconstructed images
├── README.md            # This file
└── your_notebook.ipynb  # Jupyter Notebook containing the project code
```


## License
This project is licensed under the MIT License.

## Acknowledgements
- PyTorch for the deep learning framework.
- Matplotlib for visualization.
- The [Character Font Images Data Set](https://archive.ics.uci.edu/ml/datasets/Character+Font+Images) from the UCI Machine Learning Repository.








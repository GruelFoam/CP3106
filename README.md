# CP3106
Everything about my CP3106. A firm representation learning aims to achieve better industry clustering.

# File struction introduction

## 1 baseline
Earliest baseline experiments.

## 2 data processing
Check and merge the firm descriptions from different sources to one csv file. Source of main dataset "merged_1197.csv" in "MyData".

## 3 tabular encoder
Tried to use CNN and linear model to compress the tabular data.

## 4 text fusion
Some experiments on different fusion and training methods.

- Folder "mid_data" stores the clustering results from different methods.

- **1_concatenate [fusion method]:** Directly concatenate three description embeddings of dimension 1536.

- **2_average [fusion method]:** Taking average of three description embeddings.

- **3_pca, 3_mvmds [train period 2]:** Using package to reduce the dimension of concatenated embeddings.

- **4_autoencoder [train period 2]:** Using audoencoder to compress embeddings after concatenation or summation.

- **5_mcca [train period1]:** Using CCA to transform embeddings so that embeddings from the same firm have a higher correlation. Then, take the average.

- **getting_returns:** Source of "returns_long.csv" in "MyData".

- **tool:** Containing tool functions.

## 5 Synthetic Data Generation
Some experiments on generating synthetic training data.

- **saved_model:** Saved autoencoder (more complex than the one trained in "text fusion")

- **ae:** A more complex autoencoder.

- **cgan:** A conditional GAN trained by the encoded descriptions from "ae". (performance is not ideal)
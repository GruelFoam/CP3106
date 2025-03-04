# CP3106
Everything about my CP3106. A firm representation learning aims to achieve better industry clustering.

# File struction introduction

## 1_baseline
Earliest baseline experiments.

## 2_data_processing
Check and merge the firm descriptions from different sources to one csv file. Source of main dataset "merged_1197.csv" in "MyData".

## 3_tabular_encoder
Tried to use CNN and linear model to compress the tabular data.

## 4_text_fusion
Some experiments on different fusion and training methods.

- Folder "mid_data" stores the clustering results from different methods.

- **1_concatenate [fusion method]:** Directly concatenate three description embeddings of dimension 1536.

- **2_average [fusion method]:** Taking average of three description embeddings.

- **3_pca, 3_mvmds [train period 2]:** Using package to reduce the dimension of concatenated embeddings.

- **4_autoencoder [train period 2]:** Using audoencoder to compress embeddings after concatenation or summation.

- **5_mcca [train period1]:** Using CCA to transform embeddings so that embeddings from the same firm have a higher correlation. Then, take the average.

- **getting_returns:** Source of "returns_long.csv" in "MyData".

- **tool:** Containing tool functions.


## 5_synthetic_data_generation
Some experiments on generating synthetic training data.

- **ae:** A more complex autoencoder.

- **cgan:** A conditional GAN trained by the encoded descriptions from "ae". (performance is not ideal)


## 6_with_neural_network
- **2_train_clasf_with_ae:** Origin embedding (1536) -> New embedding (256) -> Encode by clasf (128)

- **4_preparing_pairs:** Origin embedding (1536) -> New embedding (256) -> Transform by a network trained by contrastive loss.

## model
- **saved_model:** Folder that stores trained models.

- **obtain_model:** Scripts that facilitate loading trained models or initializing a blank model before training.

- Other .py files are the definition of my models.
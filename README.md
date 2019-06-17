# Grab AI Challenge 2019

This repository contains my solution for Grab AI Challenge 2019. For more information, refer to https://www.aiforsea.com or `docs/info/challenge_info.pdf`.

## 1. How to use
#### 1.1. Setup
```
conda create -n grab_ai_challenge python=3.6
source activate grab_ai_challenge
pip install -r requirements.txt
```
#### 1.2 Training
To train model, run:
```
python main_train.py --config_path <INSERT_PATH>
```
For example, to train a LSTM model, modify the parameters in `configs/lstm.yaml` and run:
```
python main_train.py --config_path src/configs/lstm.yaml
```

#### 1.3 Inference
To perform inference, run:
```
python main_inference.py --config_path <INSERT_YAML_PATH> --model_dir <INSERT_MODEL_DIR> --inference_data_path <INSERT_CSV_PATH>
```
For example, to use a trained LightGBM to perform inference, run:
```
python main_inference.py --config_path src/output/2019-06-17_18-13-13-153/config.yaml --model_dir src/output/2019-06-17_18-13-13-153/model/ --inference_data_path src/data_loader/data/sample_test.csv
```

## 2. Methodology

#### 2.1 Feature Enegineering
Some features extracted are:
- Historical demands
    - Controlled by `num_steps`
- Day
    - Converted to one hot
    - Controlled by `use_day`
- Cyclical timestamp
    - Use sine and cosine functions to transform timestamp into cyclical features
    - Controlled by `use_cyclical_timestamp`
- Part of day
    - Divide the timestamp of entire day into multiple categories, such as morning, midday, afternoon, evening, night and midnight
    - Controlled by `use_part_of_day`
- Geohash
    - Controlled by `use_geohash`

#### 2.2 Model
Instead of spending most of the time in using ensemble methods to squeeze out some improvement, we try to experiment with different models to analyze their performance, and implement some of the latest research work in deep learning (such as SpatioTCN). The models implemented are as follows:

- LightGBM
- Multilayer Perceptron (MLP)
- Long Short Term Memory (LSTM)
- Temporal Convolutional Network (TCN)
    - With dilations, causal network and skip connections
- Spatio Temporal Convolutional Network (SpatioTCN)
    - It is similar with TCN, but with an additional graph propagation layer added between each TCN block for message passing across different nodes. The graph propagation layers used here is similar to graph convolutional network [1].
    - We calculate the L2 distance among each geohash pairs and normalize them using `negative_softmax` to construct an adjacency matrix. This matrix is then fed into the graph propagation layer of SpatioTCN for message passing. Theoretically, this model should perform better than TCN as it distributes the information across geohash pairs when predicting for the geohash of interest.

## 3. Implementation
#### 3.1 Modules and Repository Structure
- Main modules
    - `data_loader`
    - `models`
    - `trainers`
- Others
    - `helpers`
    - `base`

#### 3.2 Good Case Practice
This subsection specifically addresses one of the criterias of the challenge, which is `Code Quality`. In particular, some good case practices are adopted here for a better structural quality of the code to enhance its maintainability and robustness, such as:

- Codes are seprated into three main modules: `data_loader`, `models`, `trainers`
- Logging is extensively used for debugging and organizing the experiments
- YAML configuration files are used to organize model parameters
- Since the codes are abstracted by inheritance and ABC, it could be easily extended. For example, to implement a new neural network architecture, one simply just needs to inherit `base.NN` and implement `_forward()` function, then create another YAML configuration file for it to run.

## 4. Results
The following results are computed as RMSE for test set:
- LightGBM
    - RMSE: 0.0295061609
    - Refer to `src/output/2019-06-17_18-13-13-153/training.log`
    - For inference, run `python main_inference.py --config_path src/output/2019-06-17_18-13-13-153/config.yaml --model_dir src/output/2019-06-17_18-13-13-153/model/ --inference_data_path <INSERT_CSV_PATH>`
- Multilayer Perceptron (MLP)
    - RMSE: 0.0803641468
    - Refer to `src/output/2019-06-17_18-16-38-675/training.log`
    - For inference, run `python main_inference.py --config_path src/output/2019-06-17_18-16-38-675/config.yaml --model_dir src/output/2019-06-17_18-16-38-675/model/epoch_4/model.meta --inference_data_path <INSERT_CSV_PATH>`
- Temporal Convolutional Network (TCN)
    - RMSE: 0.0395143114
    - Refer to `src/output/2019-06-17_14-20-52-632/training.log`
    - For inference, run `python main_inference.py --config_path src/output/2019-06-17_14-20-52-632/config.yaml --model_dir src/output/2019-06-17_14-20-52-632/model/epoch_13/model.meta --inference_data_path <INSERT_CSV_PATH>`
- Spatio Temporal Convolutional Network (SpatioTCN)
    - RMSE: 0.0837173931
    - Refer to `src/output/2019-06-17_21-48-17-390/training.log`
    - For inference, run `python main_inference.py --config_path src/output/2019-06-17_21-48-17-390/config.yaml --model_dir src/output/2019-06-17_21-48-17-390/model/epoch_13/model.meta --inference_data_path <INSERT_CSV_PATH>`

We did not manage to experiment with LSTM as it requires much computational resources and time to train.

It is observed that LightGBM has the best performance among all models. Theoretically, given such large amount of data, TCN and SpatioTCN should perform better than LightGBM. However, due to lack of computational resources, it is unfortunate that we aren't able to fine-tune these models for better RMSE. We also observe that SpatioTCN requires much hyperparameter tuning due to its instability throughout the training process, which might be possibly due to the graph propagation layer added.

## 5. Future Work
- Experiment with LSTM
- Fine tune TCN and SpatioTCN

## References
[1] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks.”

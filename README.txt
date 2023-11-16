1. 'data_process_gat.py' is used for data preprocessing and factors addition in 'train_data.csv'.

2. 'GAT_stock.py' is a training script for predicting stock returns based on graph attention networks (GAT), 
in which 'gat.py' (used to construct the network structure in GAT) and 'stock_dataset_dwt.py' (used to process 
training and testing data) are called. Finally, it can obtain the model 'net_params_1999.pth'.

3. 'stock_db4.npy' and 'stock_features.npy' are used in 'GAT_stock.py', They represent the correlation adjacency 
matrix between stocks and the characteristic matrix of each stock, both of them are input variables for GAT. As 
for how they were obtained, there is a detailed description in the report.

4. 'train_GAT.py' is the script for PPO training. We use adapted script 'env_stock_reward.py' as the stock trading 
environment and the processed dataset 'train_data_update.csv' as the input dataset. Finally, it can obtain the model 
'trained_ppo.zip' in 'trained_models'.

5. 'gat_ppo_test.py' is used for the testing of the GAT-PPO model, which includes the entire process of converting 
'train_data.csv' to 'train_data_update.csv'.
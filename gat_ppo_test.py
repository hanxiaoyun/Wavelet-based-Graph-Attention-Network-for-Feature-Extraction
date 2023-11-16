

import argparse
from finrl.meta.preprocessor.preprocessors import data_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stock_dataset_dwt import LoadData
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gat import GATNet
from env_stock_reward import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from finrl.main import check_and_make_directories
from finrl.plot import backtest_stats
INDICATORS=['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
       'close_30_sma', 'close_60_sma', 'return_today', 'bias', 'vroc', 'wr',
       'return_yesterday','prior']
# Contestants are welcome to split the data in their own way for model tuning
TRADE_START_DATE = '2014-01-01'
TRADE_END_DATE = '2023-10-24'
FILE_PATH = 'train_data.csv'
FILE_PATH2= 'train_data_update.csv'


def bias(x):
    x['bias'] = (x.close - x['close'].rolling(window=5).mean()) / x['close'].rolling(window=5).mean()
    return x

def vroc(x):
    x['vroc'] = (x['volume'] - x['volume'].shift(5)) / x['volume'].shift(5)
    return x

def wr(x):
    x['wr'] = (x['high'].rolling(5).max() - x['close']) / (x['high'].rolling(5).max() - x['low'].rolling(5).min())
    return x

def return_yesterday(x):
    x['return_yesterday'] = (x['close'] - x['close'].shift(1)) / x['close'].shift(1)
    return x

def z_score(dt):
    dt1 = dt.iloc[:, :2]
    dtclose = dt.iloc[:, 5]
    dt2 = dt.iloc[:, 2:]
    mean = dt2.mean()
    std = dt2.std()
    dt2 = (dt2 - mean) / std
    dt2.rename(columns={'close': 'close_z'}, inplace=True)
    dt = pd.concat([dt1, dtclose, dt2], axis=1)
    return dt

def gat_test():

    test_data = LoadData(data_path=["./stock_db4.npy", "./stock_features.npy"], num_nodes=29, divide_days=[7,3345],
                         time_interval=1, history_length=7,
                          train_mode="test")

    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

    my_net = GATNet(in_c=7*1 , hid_c=32, out_c=1, n_heads=2)

    device = torch.device("cpu")

    my_net = my_net.to(device)

    criterion = nn.MSELoss()

    state_dict=torch.load('./net_params_1999.pth')

    my_net.load_state_dict(state_dict)
    my_net.eval()

    with torch.no_grad():


        Predict = np.zeros([29, 1, 1])

        total_loss = 0.0
        for data in test_loader:

            predict_value = my_net(data, device)

            loss = criterion(predict_value, data["flow_y"])

            total_loss += loss.item()
            predict_value = predict_value.transpose(0, 2).squeeze(0)

            data_to_save = compute_performance(predict_value,  test_loader)
            Predict = np.concatenate([Predict, data_to_save[0]], axis=1)


    Predict = np.delete(Predict, 0, axis=1)
    pre=pd.DataFrame(np.squeeze(Predict,axis=2))
    return pre

def compute_performance(prediction,  data):
    try:
        dataset = data.dataset
    except:
        dataset = data
    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy(),'pre')

    recovered_data = [prediction]

    return  recovered_data




# PPO configs
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}


if __name__ == '__main__':
    # We will use unseen, post-deadline data for testing
    TRAINED_MODEL_DIR = 'trained_models'
    parser = argparse.ArgumentParser(description='Description of program')
    parser.add_argument('--start_date', default=TRADE_START_DATE, help='Trade start date (default: {})'.format(TRADE_START_DATE))
    parser.add_argument('--end_date', default=TRADE_END_DATE, help='Trade end date (default: {})'.format(TRADE_END_DATE))
    parser.add_argument('--data_file', default=FILE_PATH, help='Trade data file')
    parser.add_argument('--data_file2', default=FILE_PATH2, help='Trade data file2')

    args = parser.parse_args()
    TRADE_START_DATE = args.start_date
    TRADE_END_DATE = args.end_date

    df = pd.read_csv(args.data_file, index_col=0)
    df.drop(columns=['day', 'vix', 'turbulence'], inplace=True)
    df.fillna(0, inplace=True)
    df['return_today'] = (df['close'] - df['open']) / df['close']
    df = df.groupby('tic').apply(lambda x: bias(x))
    df.reset_index(drop=True,inplace=True)
    df = df.groupby('tic').apply(lambda x: vroc(x))
    df.reset_index(drop=True, inplace=True)
    df = df.groupby('tic').apply(lambda x: wr(x))
    df.reset_index(drop=True, inplace=True)
    df = df.groupby('tic').apply(lambda x: return_yesterday(x))
    df.reset_index(drop=True, inplace=True)
    df = df.groupby('date').apply(lambda x: z_score(x))
    df.reset_index(drop=True, inplace=True)
    df=df.sort_values(['date','tic'])
    df.fillna(0, inplace=True)
    df.to_csv(args.data_file2)
    df = df.drop(columns=['tic','close'])
    date = df.date.unique()
    a = np.zeros(shape=[len(date), 29, 18])
    for i in range(len(date)):
        a[i] = df[df['date'] == date[i]].drop(columns=['date']).values
    np.save('stock_features.npy', a)

    pre=gat_test()
    df = pd.read_csv(args.data_file2, index_col=0)
    x1 = pre.T.values.flatten().tolist()
    x0 = [0 for _ in range(7 * 29)]
    df['prior'] = x0 + x1
    df.to_csv(args.data_file2)

    processed_full = pd.read_csv(args.data_file2)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    # please do not change initial_amount
    env_kwargs = {
        "hmax": 160,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    check_and_make_directories([TRAINED_MODEL_DIR])

    # Environment
    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)
    
    # PPO agent
    agent = DRLAgent(env = e_trade_gym)
    model_ppo = agent.get_model("ppo", model_kwargs = PPO_PARAMS)
    trained_ppo = PPO.load(TRAINED_MODEL_DIR + '/trained_ppo')

    # Backtesting
    df_result_ppo, df_actions_ppo = DRLAgent.DRL_prediction(model=trained_ppo, environment = e_trade_gym)

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(account_value=df_result_ppo)
    
    """Plotting"""
    plt.rcParams["figure.figsize"] = (15,5)
    plt.figure()
    
    df_result_ppo.plot()
    plt.savefig("plot.png")
    
    df_result_ppo.to_csv("results.csv", index=False)

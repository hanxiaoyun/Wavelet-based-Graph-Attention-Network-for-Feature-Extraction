import pandas as pd
from finrl.meta.preprocessor.preprocessors import data_split
import torch
from env_stock_reward import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories


#INDICATORS=['macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30','close_30_sma','close_60_sma']
INDICATORS=['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
       'close_30_sma', 'close_60_sma', 'return_today', 'bias', 'vroc', 'wr',
       'return_yesterday','prior']
# Contestants are welcome to split the data in their own way for model tuning
TRAIN_START_DATE = '2010-07-01'
TRAIN_END_DATE = '2023-10-24'

processed_full = pd.read_csv('./train_date_update.csv')
train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)

# Environment configs
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 50,
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

# PPO configs
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "batch_size": 128,
}


if __name__ == '__main__':
    torch.set_num_threads(5)
    TRAINED_MODEL_DIR='./trained_models'
    RESULTS_DIR='./results'
    check_and_make_directories([TRAINED_MODEL_DIR])

    # Environment
    e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    # PPO agent
    agent = DRLAgent(env = env_train)
    model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

    # set up logger
    tmp_path = RESULTS_DIR + '/ppo'
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)

    trained_ppo = agent.train_model(model=model_ppo,
                                tb_log_name='ppo',
                                total_timesteps=5500000)
    
    trained_ppo.save(TRAINED_MODEL_DIR + '/trained_ppo')

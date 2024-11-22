import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# 금융 데이터 다운로드 (예: S&P 500 구성종목)
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2010-01-01", end="2020-01-01")["Adj Close"]

# 로그 수익률 계산
returns = np.log(data / data.shift(1)).dropna()

# 마코위츠 최적화 포트폴리오 비중 계산 함수
def markowitz_optimization(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    def portfolio_annual_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return returns, std
    
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
        p_returns, p_std = portfolio_annual_performance(weights, mean_returns, cov_matrix)
        return -(p_returns - risk_free_rate) / p_std
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

markowitz_weights = markowitz_optimization(returns)
print("Markowitz Optimal Weights:", markowitz_weights)

# 온라인 학습이 가능한 강화학습 환경 구성
class OnlineLearningPortfolioEnv(gym.Env):
    def __init__(self, returns, markowitz_weights=None, initial_balance=1000, ppo_model=None):
        super(OnlineLearningPortfolioEnv, self).__init__()
        self.returns = returns
        self.num_assets = returns.shape[1]
        self.initial_balance = initial_balance
        self.markowitz_weights = markowitz_weights if markowitz_weights is not None else np.ones(self.num_assets) / self.num_assets
        self.ppo_model = ppo_model

        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1 + self.num_assets + self.num_assets,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio = self.markowitz_weights
        self.ppo_weight = 0.5  # 초기 가중치 설정
        return self._get_observation()

    def step(self, action):
        if self.current_step >= len(self.returns) - 1:
            done = True
            return self._get_observation(), self.balance, done, {}

        action = np.clip(action, -0.1, 0.1)
        ppo_portfolio = np.clip(self.portfolio + action, 0, 1)
        ppo_portfolio = ppo_portfolio / np.sum(ppo_portfolio)

        self.ppo_weight = min(1.0, self.ppo_weight + 0.01)
        self.portfolio = (1 - self.ppo_weight) * self.markowitz_weights + self.ppo_weight * ppo_portfolio

        self.current_step += 1
        rewards = np.dot(self.portfolio, self.returns.iloc[self.current_step])
        self.balance *= (1 + rewards)

        if self.ppo_model:
            obs = self._get_observation()
            action_reshaped = action.reshape((1, -1))
            reward_reshaped = np.array([rewards]).reshape((1,))
            self.ppo_model.learn(total_timesteps=1, log_interval=1)

        done = self.current_step == len(self.returns) - 1
        return self._get_observation(), self.balance, done, {}

    def _get_observation(self):
        if self.current_step < len(self.returns):
            obs = np.concatenate(([self.balance], self.portfolio, self.returns.iloc[self.current_step].values))
        else:
            obs = np.concatenate(([self.balance], self.portfolio, np.zeros(self.num_assets)))
        return np.array(obs, dtype=np.float32)

# PPO 모델 및 새로운 환경 구성
online_ensemble_env = DummyVecEnv([lambda: OnlineLearningPortfolioEnv(returns, markowitz_weights)])
online_ensemble_model = PPO('MlpPolicy', online_ensemble_env, verbose=1)

# 백테스트 함수
def backtest(env, model):
    state = env.reset()
    balances = [env.envs[0].initial_balance]
    for t in range(len(env.envs[0].returns) - 1):
        action, _ = model.predict(state)
        next_state, balance, done, _ = env.step(action)
        balances.append(balance[0])
        state = next_state
        if done:
            break
    return balances

# 백테스트 및 성능 평가
online_balances = backtest(online_ensemble_env, online_ensemble_model)

def calculate_performance(balances):
    returns = np.diff(balances) / balances[:-1]
    annual_return = np.mean(returns) * 252
    annual_volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility
    return annual_return, annual_volatility, sharpe_ratio

# 마코위츠 포트폴리오 성능 계산
def backtest_markowitz(returns, weights, initial_balance=1000):
    balances = [initial_balance]
    for t in range(1, len(returns)):
        rewards = np.dot(weights, returns.iloc[t])
        new_balance = balances[-1] * (1 + rewards)
        balances.append(new_balance)
    return balances

markowitz_balances = backtest_markowitz(returns, markowitz_weights)

# 성능 비교
online_annual_return, online_annual_volatility, online_sharpe_ratio = calculate_performance(online_balances)
print("Online Ensemble Portfolio Performance:")
print(f"Annual Return: {online_annual_return:.2f}")
print(f"Annual Volatility: {online_annual_volatility:.2f}")
print(f"Sharpe Ratio: {online_sharpe_ratio:.2f}")

# 결과 시각화
plt.plot(online_balances, label="Online Ensemble Portfolio")
plt.plot(markowitz_balances, label="Markowitz Portfolio")
plt.title("Portfolio Performance Comparison")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()

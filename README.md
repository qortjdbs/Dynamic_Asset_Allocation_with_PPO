# Dynamic Asset Allocation using Markowitz Theory and Proximal Policy Optimization (PPO)

This project integrates traditional Markowitz portfolio theory with reinforcement learning, specifically the Proximal Policy Optimization (PPO) algorithm, to create a dynamic asset allocation strategy. The hybrid model leverages Markowitz’s mean-variance optimization for stability and PPO’s adaptability to changing market conditions.

## Key Features:
- **Markowitz Theory**: Implements Mean-Variance Optimization (MVO) to determine stable asset distribution.
- **Proximal Policy Optimization (PPO)**: Uses reinforcement learning with PPO for dynamic portfolio adjustments, ensuring adaptability to market fluctuations.
- **Hybrid Approach**: Combines the benefits of Markowitz’s theory for initial stability and PPO for continuous improvement over time.
- **Performance**: The hybrid model achieved a cumulative return of 168% compared to 336% using PPO alone and 98% with only Markowitz theory in a one-year backtest.
- **Risk Management**: Improved risk metrics (Sharpe ratio and Sortino ratio) demonstrate effective risk management alongside high returns.

## Methodology:
- **Reinforcement Learning Model**: Adjusts asset distribution between -10% and 10% at each time step, with rewards tied to portfolio performance.
- **Markowitz-PPO Integration**: Combines Markowitz’s asset distribution weights with reinforcement learning's dynamic adjustments to mitigate the early-stage risks of PPO.
- **Backtesting**: Conducted with actual S&P 500 market data, comparing the performance of three portfolios: Markowitz-only, PPO-only, and the hybrid approach.

## Results:
- **Stable Returns**: The hybrid portfolio demonstrated high performance from the start, effectively balancing between stability and adaptability.
- **Superior Risk-Adjusted Returns**: Metrics such as the Sharpe and Sortino ratios were higher compared to the individual strategies.

## Conclusion:
This study presents a novel approach to asset allocation by integrating traditional portfolio theory with reinforcement learning, achieving better performance and risk management than using either strategy alone.

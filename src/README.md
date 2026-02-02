# Pistonball Multi-Agent Training

基于 PettingZoo Pistonball 环境的多智能体训练框架，支持 LLM 策略和传统 RL 策略。

## 安装依赖

```bash
pip install pettingzoo[butterfly] supersuit imageio numpy
# 如果使用 LLM 策略
pip install openai textgrad
```

## 快速开始

### 1. 运行单个 Episode 并生成 GIF

```python
from training.episode_generator import EpisodeGenerator

# 创建生成器
generator = EpisodeGenerator(
    num_pistons=20,          # 活塞数量
    max_cycles=125,          # 最大步数
    action_mode="discrete",  # "discrete" | "fine" | "continuous"
    agent_type="rule",       # "rule" | "llm"
)

# 运行并渲染单个 episode
reward, steps, gif_path = generator.render_single_episode(
    save_path="./my_episode.gif",
    fps=30
)
print(f"Reward: {reward}, Steps: {steps}, GIF: {gif_path}")
```

### 2. 生成多个 Episodes

```python
# 生成多个 episodes（带 GIF）
episodes = generator.generate_episodes(
    num_episodes=5,
    iteration=0,
    save_gif=True,
    gif_fps=30,
)

for ep in episodes:
    print(f"Episode {ep['episode_id']}: reward={ep['total_reward']:.2f}")
```

### 3. 评估策略

```python
# 评估策略性能
eval_results = generator.evaluate_policy(
    policy_path="./policy.json",  # 可选，使用默认策略则为 None
    num_episodes=10,
    save_dir="./eval_results",
)

print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
```

## 动作空间

支持三种动作模式：

### Discrete (3 actions)
```
0: push_up      - 向上推动活塞
1: stay         - 保持不动
2: retract_down - 向下收缩
```

### Fine-grained (7 actions)
```
0: push_up_fast     (+1.0)  - 最大速度推
1: push_up_medium   (+0.6)  - 中等速度推
2: push_up_slow     (+0.3)  - 缓慢推
3: stay             (0.0)   - 保持不动
4: retract_slow     (-0.3)  - 缓慢收缩
5: retract_medium   (-0.6)  - 中等速度收缩
6: retract_fast     (-1.0)  - 最大速度收缩
```

### Continuous
```
[-1.0, 1.0] 连续值
+1.0 = 最大推力向上
 0.0 = 保持不动
-1.0 = 最大收缩
```

## 观测空间 (CTDE 架构)

### Actor (局部观测 - 分散执行)
每个 agent 只能看到自己的局部信息：
```
=== Your Piston Status ===
Piston index: #2 of 20 (center)
Your X position: 125.0

=== Ball Relative to You ===
Ball distance: 73.0 pixels - NEARBY
Ball is to your RIGHT (coming towards you)
```

### Critic (全局观测 - 集中训练)
Critic 可以访问全局状态用于计算 loss：
```python
transition["global_state"] = {
    "text": "...",                    # 全局状态文本
    "ball_position": [159.0, 416.0], # 球的精确坐标
    "ball_velocity": [0.0, 0.0],     # 球的速度
    "pistons": [...],                 # 所有活塞状态
}
```

## 输出文件结构

```
experiments/
└── exp_name/
    └── iteration_0/
        └── episode_0/
            ├── episode_0.gif      # 渲染动画
            ├── episode_log.txt    # 详细文本日志
            ├── episode.json       # 结构化数据
            └── trajectory.pkl     # Pickle 轨迹数据
```

## 使用 TextGrad 进行策略优化

```python
from training.trainer import MonteCarloTrainer
from training.config import TrainingConfig

config = TrainingConfig(
    paradigm="central_credit",  # "independent" | "central_global" | "central_credit"
    num_pistons=20,
    num_iterations=10,
    trajectories_per_iteration=5,
    model="gpt-4o-mini",
)

trainer = MonteCarloTrainer(config)
stats = trainer.train_full()
```

## 训练范式

| 范式 | 描述 |
|-----|------|
| `independent` | 每组活塞独立评估和优化 |
| `central_global` | 全局团队奖励，共享策略 |
| `central_credit` | 信用分配，评估各组贡献 |

## API 参考

### EpisodeGenerator

```python
EpisodeGenerator(
    num_pistons: int = 20,
    max_cycles: int = 125,
    frame_size: Tuple[int, int] = (64, 64),
    stack_size: int = 4,
    continuous: bool = False,
    experiments_dir: Path = "./experiments",
    exp_name: str = "pistonball_exp",
    agent_type: str = "rule",      # "rule" | "llm"
    gpt_model: str = "gpt-4o-mini",
    action_mode: str = "discrete", # "discrete" | "fine" | "continuous"
)
```

### 主要方法

| 方法 | 描述 |
|-----|------|
| `generate_episodes()` | 生成多个 episodes |
| `evaluate_policy()` | 评估策略性能 |
| `render_single_episode()` | 渲染单个 episode 为 GIF |

### PistonballObservationFormatter

| 方法 | 描述 |
|-----|------|
| `get_env_state(env)` | 获取环境内部状态 |
| `format_local_observation(env, agent_idx)` | 格式化局部观测 |
| `format_global_state(env)` | 格式化全局状态 |
| `format_agent_prompt(env, agent_name, policy, action_mode)` | 生成 LLM prompt |

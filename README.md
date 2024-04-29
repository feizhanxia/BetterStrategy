# BetterStrategy

 毕业论文的, 仿真实验/计算分析的, 代码/文档

## Files

```
BetterStrategy/
│
├── environment/               # 与环境相关的代码
│   ├── __init__.py            # 使environment目录可作为模块
│   └── env.py                 # 定义环境的类和逻辑
│
├── agent/                     # 包含与agent相关的所有代码
│   ├── __init__.py            # 使agent目录可作为模块
│   └── prey.py                # 定义猎物agent的类和行为
│
├── utils/                     # 通用工具和辅助函数
│   ├── __init__.py            # 使utils目录可作为模块
│   ├── sampling.py            # 初始化时采样工具代码
│   └── kinematics.py          # 包含运动学模型和相关计算
│
├── params/                    # 与决策模型相关的代码
│   ├── env_configs.yaml       # 环境的配置参数文件
│   └── train_configs.yaml     # 训练过程的超参数配置文件
│
├── output/                    # 存储输出和训练结果
│   ├── best_model/            # 存储选出的代表性的模型权重
│   ├── checkpoints/           # 每轮训练过程中间自动保存的权重
│   ├── ppo_tensorboard/       # 存储训练日志文件（tensorboard）
│   └── videos/                # 存储训练好模型的测试渲染视频
│
├── check.py                   # 环境检查代码
├── train.py                   # 模型训练代码
├── enjoy.py                   # 模型测试代码
├── test-gpu-speed.py          # 使用gpu与cpu训练的运行速度对比
├── test-step-time.py          # 平均每步运行的耗时
├── Dockerfile                 # 代码运行环境容器配置文件
├── requirements.txt           # 项目依赖文件
└── README.md                  # 项目说明文件
```

## Usage

First, `pip install --upgrade pip setuptools wheel`

Second, `pip install -r requirements.txt`

Then, run any code here for your specific purpose.

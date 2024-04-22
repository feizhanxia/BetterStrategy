# BetterStrategy

 毕业论文的, 仿真实验/计算分析的, 代码/文档

```
BetterStrategy/
│
├── output/                    # 存储输出和训练结果
│   ├── models/                # 存储训练好的模型权重
│   ├── logs/                  # 存储训练日志文件
│   └── visualizations/        # 存储生成的可视化文件
│
├── agent/                     # 包含与agent相关的所有代码
│   ├── __init__.py            # 使agent目录可作为模块
│   ├── predator_agent.py      # 定义捕食者agent的类和行为
│   └── prey_agent.py          # 定义猎物agent的类和行为
│
├── environment/               # 与环境相关的代码
│   ├── __init__.py            # 使environment目录可作为模块
│   └── env.py                 # 定义环境的类和逻辑
│
├── model/                     # 与决策模型相关的代码
│   ├── __init__.py            # 使model目录可作为模块
│   └── decision_model.py      # 定义决策模型的类和函数
│
├── utils/                     # 通用工具和辅助函数
│   ├── __init__.py            # 使utils目录可作为模块
│   └── kinematics.py          # 包含运动学模型和相关计算
│
├── main.py                    # 项目的主执行文件,用于启动和运行模拟
├── requirements.txt           # 项目依赖文件
└── README.md                  # 项目说明文件
```

First, `pip install --upgrade pip setuptools wheel`

Second, `pip install -r requirements.txt`

Then, `python main.py`

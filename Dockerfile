# 使用兼容 Apple Silicon 的官方 Python 镜像
FROM arm64v8/python:3.10-buster

# 设置工作目录
WORKDIR /workspace

# 更新 pip 和安装基本 Python 工具
RUN python -m pip install --upgrade pip setuptools wheel

# 安装科学计算和可视化库
RUN pip install numpy scipy matplotlib pandas seaborn

# 安装 PyTorch 和相关库,从 PyTorch 官方的夜间构建中获得 Apple MPS 支持
RUN pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# 安装强化学习库和 Jupyter
RUN pip install gymnasium stable-baselines3 jupyter

# 配置容器默认启动命令
CMD ["bash"]
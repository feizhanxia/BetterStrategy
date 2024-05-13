# 初始化环境并运行
obs, info = env.reset()
done = False
while not done:
    # 预测动作
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _truncated, info = env.step(action)


# 创建回调函数, 用于记录每一次捕获的步数
class RecordStepCapture(BaseCallback):
    def __init__(self, config_id: str, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.config_id = config_id
        self.data = []
        self.save_path = save_path 

    def _on_training_start(self):
        self.n_capture = np.zeros(self.training_env.n_envs, dtype=int)
        self.n_steps = np.zeros(self.training_env.n_envs, dtype=int)
        self.n_episode = np.arange(1, self.training_env.n_envs+1)

    def _on_step(self) -> bool:
        infos = self.training_env.get_attr('infos')
        dones = self.training_env.get_attr('dones')
        for i in range(self.training_env.n_envs):
            if infos[i]['capture_count']>self.n_capture[i]:
                self.n_capture[i] = infos[i]['capture_count']
                self.n_steps[i] = infos[i]['num_steps']
                # 记录数据
                self.data.append({
                    'config_id': self.config_id,
                    'episode': self.n_episode[i],
                    'n_capture': self.n_capture[i],
                    'n_steps': self.n_steps[i]
                })
            if dones[i]:
                self.n_capture[i] = 0
                self.n_steps[i] = 0
                self.n_episode[i] = np.max(self.n_episode) + 1
        return True
    
    def _on_training_end(self):
        # 创建 DataFrame 并保存为 CSV
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path, index=False)
        print(f"Data saved to {self.save_path}")

record_step_callback = RecordStepCapture(config_id, save_path)
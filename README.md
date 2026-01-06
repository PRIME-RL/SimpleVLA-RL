<div align="center">

<img src="figs/logo.png" width="260"/>

## SimpleVLA-RL: Open RL Framework for Vision–Language–Action Models

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674) [![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

<!-- <div align="center">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#overview" style="text-decoration: none; font-weight: bold;">📖 Overview</a> •
    <a href="#main-results" style="text-decoration: none; font-weight: bold;">📃 Main Results</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a>
  </p>
  <p>
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">🌻 Acknowledgement</a> •
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📨 Contact</a> •
    <a href="#todo" style="text-decoration: none; font-weight: bold;">📝 TODO</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div> -->
**SimpleVLA-RL** is an efficient RL framework for VLA that improves long-horizon planning under data scarcity. It leverages reinforcement learning that can substantially outperforms SFT in simulation and real-world tasks, reveals a "pushcut" new-action phenomenon, and strengthens spatial/object/goal generalization.
</div>

# 🎉News
- **[2026-01-01]** **Building upon $${\color{red}\textbf{SimpleVLA-RL}}$$, we have implemented $${\color{red}\textbf{real-world RL on long-horizon dexterous tasks}}$$  and witnessed a $${\color{red}\textbf{non-trivial}}$$ (~relatively 300\%)  performance improvement over the SFT model, along with surprising capabilities on auto-recovery. Blog coming soon.**
- **[2025-10-01]** **SimpleVLA-RL** now supports RoboTwin2.0 Benchmark. Feel free to experiment with it!
- **[2025-09-12]** Excited to release the **SimpleVLA-RL** paper! Check it out: [Paper](https://arxiv.org/abs/2509.09674).
- **[2025-05-27]** We release the code of **SimpleVLA-RL**.

</div>

# 📌Highlights

#### Efficient and Effective VLA Reinforcement Learning Framework
- End-to-end VLA RL pipeline built on [veRL](https://github.com/volcengine/verl) with VLA-specific optimizations
- Multi-environment parallel rendering significantly accelerates VLA trajectory sampling
- Leverages [veRL](https://github.com/volcengine/verl)'s state-of-the-art infrastructure: efficient distributed training (FSDP), hybrid communication patterns, and optimized memory management for fast training/inference

#### Model and Environment Support
- **VLA Models**: [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- **Benchmarks**: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [RoboTwin 1.0/2.0](https://github.com/TianxingChen/RoboTwin)
- Modular architecture for easy integration of new VLA models, benchmarks and RL algorithms (Upcoming)

#### Minimal Reward Engineering and Exploration Strategies
- Binary (0/1) outcome rewards - no complex reward design needed
- Exploration strategies: dynamic sampling, adaptive clipping, temperature tuning
</div>

# 🔧Key Implementations

SimpleVLA-RL extends veRL with VLA-specific components across the following modules:

**[`verl/trainer/main_ppo.py`](verl/trainer/main_ppo.py)**
- Main entry point with ray initialization
- `RobRewardManager` for reward distribution

**[`verl/trainer/ppo/ray_trainer.py`](verl/trainer/ppo/ray_trainer.py)**
- Main RL training loop: data loading, VLA rollout, model updates, evaluation, checkpointing
- RL algorithm-specific advantage computation

**[`verl/workers/fsdp_workers.py`](verl/workers/fsdp_workers.py)**
- Source of core functions called in `ray_trainer.py`
- VLA model/optimizer initialization, `generate_sequences`, `compute_entropy`, `update_actor`

**[`verl/workers/actor/dp_rob.py`](verl/workers/actor/dp_rob.py)**
- Specific implementation of functions in `fsdp_workers.py`
- RL loss computation, policy updates, `compute_log_prob`, `compute_entropy`

**[`verl/workers/rollout/rob_rollout.py`](verl/workers/rollout/rob_rollout.py)**
- VLA rollout implementation: environment creation, multi-environment parallel rendering, VLA action generation, environment interaction, video saving, trajectory and 0/1 reward collection

**[`verl/utils/dataset/rob_dataset.py`](verl/utils/dataset/rob_dataset.py)**
- Dataset construction for training/testing across benchmarks

**[`verl/utils/vla_utils/`](verl/utils/vla_utils/)**
- VLA model implementations (OpenVLA-OFT/OpenVLA from official code)
</div>

# ✨Getting Started

#### 1. Set Up the Environment

See [SETUP.md](SETUP.md) for detailed instructions on setting up the conda environment.  

#### 2. Prepare the SFT Model

An **SFT (Supervised Fine-Tuning)** VLA model is required for RL training. Below are the available options:

* **OpenVLA-OFT SFT Models**  
  Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models include:
  - `libero-10 traj1/trajall SFT`
  - `libero-goal/object/spatial traj1 SFT`
  - `Robotwin2.0 tasks traj1000 SFT`
* **OpenVLA SFT Models**  
  Download from [here](https://huggingface.co/openvla).

* **Other Models**  
  For other models, you may need to fine-tune them yourself.

#### 3. Train with SimpleVLA-RL

Before running the training script, ensure the following configurations are properly set:

- **Set Your Weights and Biases (WandB) API Key**  
   Replace the `WANDB_API_KEY` field in `SimpleVLA-RL/align.json` with your own WandB API key.

- **Modify Key Variables**  
   Update the following variables in `examples/run_openvla_oft_rl_libero/twin2.sh` as needed:
  - `WANDB_API_KEY`: Your WandB API key.
  - `EXPERIMENT_NAME`: The name of your experiment. You can choose any name.
  - `SFT_MODEL_PATH`: Path to your SFT model.
  - `CKPT_PATH`: Path where your checkpoints will be saved.
  - `DATASET_NAME`: For detailed options, refer to `examples/run_openvla_oft_rl_libero/twin2.sh`.
  - `ALIGN_PATH`: Path to the `SimpleVLA-RL/align.json` file.
  - `NUM_GPUS`: Number of GPUs available per node (e.g., `8`).
  - `NUM_NODES`: Number of nodes used for RL training (e.g., `1`).

> [!NOTE]
> 
> - The script has been tested on the following configurations:
>   - Single-node setup: `NUM_NODES=1`, `NUM_GPUS=8` (1 node with 8 NVIDIA A800 GPUs, each having 80GB memory).
>   - Multi-node setup: `NUM_NODES=2`, `NUM_GPUS=8` (2 nodes with 16 NVIDIA A800 GPUs, each having 80GB memory).
> - The driver version used is `470.161.03`, and the CUDA version is `12.4`. *(Not necessary)*

- **Run RL Training**  
   Use the following command to start RL training for OpenVLA-OFT on the LIBERO or RoboTwin2.0 benchmark:
  
  ```bash
  bash examples/run_openvla_oft_rl_libero.sh
  or
  bash examples/run_openvla_oft_rl_twin2.sh
  ```
  

#### 4. Run Evaluation

To evaluate the performance of your model, enable evaluation mode by setting `trainer.val_only=True` in `examples/run_openvla_oft_rl_libero/twin2.sh`. Then, execute the same script:

```bash
bash examples/run_openvla_oft_rl_libero.sh
or
bash examples/run_openvla_oft_rl_twin2.sh
```

# 📃 Main Results

We evaluate SimpleVLA-RL on the LIBERO using OpenVLA-OFT. SimpleVLA-RL improves the performance of OpenVLA-OFT to 97.6 points on LIBERO-Long and sets a new state-of-the-art. Remarkably, using only one trajectory per task for cold-start SFT, SimpleVLA-RL raises the performance of OpenVLA-OFT from 17.3 to 91.7, yielding an improvement of 74.4 points (430.1%).

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

<div align="center">
<img src="figs/teaser.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

# 🌻Acknowledgement

We develop this preview version of the code based on [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin.git), and [PRIME](https://github.com/PRIME-RL/PRIME). We acknowledge their significant contributions!
For further details and updates, please refer to the official documentation and repositories of the respective projects.
</div>


</div>

# 📝Roadmap

### Expanding Model Support
- [ ] Support advanced diffusion based RL: [pi0](https://github.com/Physical-Intelligence/openpi) and [pi0.5](https://github.com/Physical-Intelligence/openpi) with flow matching RL
- [ ] Support more VLA models: especially for lightweight ones (e.g. [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter), [SmolVLA](https://huggingface.co/docs/lerobot/smolvla))

### Expanding Environment Support
- [ ] Support more benchmarks: e.g. [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [BEHAVIOR](https://github.com/behavior-robot-suite/brs-algo), [Calvin](https://github.com/mees/calvin)
- [ ] Support real-world RL.

### Expanding Framework
- [ ] Additional online RL methods and Offline RL algorithms
- [ ] Modular environment and VLA interface for easy adaptation
- [ ] Further optimize the RL framework to achieve more efficient training
</div>

# 📨Contact

- Haozhan Li: zhan72426@gmail.com
- Ning Ding: dingning@mail.tsinghua.edu.cn

# 🎈Citation

If you find SimpleVLA-RL helpful, please cite us:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```

# 🌟Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PRIME-RL/SimpleVLA-RL&type=Date)](https://www.star-history.com/#PRIME-RL/SimpleVLA-RL&Date)

# 🚗 MountainCar Q-Learning AI

An implementation of a Reinforcement Learning agent that solves the **MountainCar-v0** environment using the **Q-learning algorithm** with custom reward shaping.

---

## 📌 Overview

This project trains an autonomous agent to solve a classic control problem where a car must reach the top of a hill despite having insufficient engine power. The agent learns to exploit momentum through trial-and-error interactions with the environment.

---

## ⚙️ Key Features

- Q-learning with discretized state space (18×18 grid)
- Custom reward shaping for faster convergence
- Epsilon-greedy exploration strategy
- Trained agent playback (no randomness)
- Q-value heatmap visualization for interpretability

---

## 🧠 How It Works

### State Space
The environment provides:
- Car position  
- Car velocity  

These continuous values are discretized into a finite grid to build a Q-table.

---

### Learning Algorithm

The agent updates its knowledge using the Q-learning update rule:

- Learns optimal actions per state  
- Balances exploration vs exploitation  
- Stores results in a Q-table  

---

### Reward Engineering (Core Idea)

Unlike standard implementations, this project introduces:

- Momentum-based rewards  
- Height-based incentives  
- Large terminal reward for reaching the goal  

This significantly improves learning efficiency and avoids stagnation.

---

## 📁 Project Structure

```
MountainCar-Qlearning/
│
├── qlearning_final.py     # Training script
├── qlearning.py           # Run trained agent
├── heatmap.py             # Visualization
├── mountain_car_brain.npy # Trained Q-table
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/mountaincar-qlearning-ai.git
cd mountaincar-qlearning-ai
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train the Agent
```bash
python qlearning_final.py
```

### 2. Run the Trained Agent
```bash
python qlearning.py
```

### 3. Visualize Learning (Heatmap)
```bash
python heatmap.py
```

---

## 📊 Results

- Agent successfully learns to reach the goal  
- Demonstrates efficient momentum control  
- Heatmap shows learned policy and confidence levels  

---

## 🛠️ Tech Stack

- Python  
- NumPy  
- Gymnasium  
- Matplotlib  

---

## 📈 Future Improvements

- Deep Q-Network (DQN) implementation  
- Continuous state handling without discretization  
- Hyperparameter optimization  
- Real-time training visualization  

---

## 🎯 Key Takeaways

This project demonstrates:

- Strong understanding of Reinforcement Learning fundamentals  
- Ability to design effective reward functions  
- Experience with simulation environments and AI training  

---

## 📬 Contact

If you have questions or suggestions, feel free to reach out.

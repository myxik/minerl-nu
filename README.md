# MineRL: Solving Minecraft with Reinforcement Learning
## General info
This repository is a source code for the course CSCI 494 - Deep Learning. The team members are
<ul>
<li>Maxim Mametkulov</li>
<li>Anuar Suleimenov</li>
<li>Nuradil Kozhakhmet</li>
<li>Abay Artykbayev</li>
</ul>

## Usage

This repository currently does not support reproducibility and it is unclear, whether it is possible(reproducible RL is tough). Currently, we provided the code for the baseline, located in our <code>minecraft</code> folder. In order to try experiments yourself, please follow the instructions
<ol>
<li>Clone this repository</li>
<li>Build a container using <code>make build</code></li>
<li>Install TurboVNC and VGL, it is located in <code>setup</code> folder using <code>sudo dpkg --install ...</code></li>
<li>Launch Turbovnc using <code>/opt/TurboVNC/bin/vncserver</code> and configure</li>
<li>Set <code>DESKTOP</code> environmental variable to :1</li>
<li>Download a dataset with a command <code>python3 -m minerl.data.download --environment "MineRLTreechop-v0"</code></li>
<li>Run agent with <code>python3 minecraft/train_agent.py</code></li>
</ol>

## Implemented Algorithms

<ol>
<li>DQN</li>
<li>ResDQN</li>
<li>Wolpertinger</li>
</ol>

## Further Research

This repository will be enlarged with different models, customizable models and different enviornments. Hopefully, reproducibility will not be an issue. In order to support us, drop us an email on how you liked organization of this repo :)

The TODO list for the nearest time is:
<ul>
<li>Write an agent that will be able to complete an episode</li>
</ul>
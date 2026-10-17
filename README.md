# Getting Over It, played by reinforcement learning

<p align="center">
  <img src="media/climb.gif" width="480" alt="The networks climbing through the boxes and the furniture">
</p>

Neural networks playing the real *Getting Over It with Bennett Foddy*, with one goal: be the first to finish it with
reinforcement learning, in one continuous climb from the bottom to the top with no save/load.

**Best run so far: 63% of the way to the top**, in one continuous game from the bottom.

## Progress by part

The climb split into the parts speedrunners use. The networks play each part alone from its entrance, 20 times, with
up to 10 minutes per try; the progress is how many tries got through to the next part (**Solved** from 90%). The
circle in the top corner is the networks' mouse input.

| Part | Progress | |
|---|---|---|
| **Coffee Rock** | Solved | <img src="media/parts/01-coffee-rock.jpg" width="360" alt="Coffee Rock"> |
| **The Devil's Chimney** | 85% | <img src="media/parts/02-the-devil-s-chimney.jpg" width="360" alt="The Devil's Chimney"> |
| **Construction Wall** | 65% | <img src="media/parts/03-construction-wall.jpg" width="360" alt="Construction Wall"> |
| **Slide Skip** | 80% | <img src="media/parts/04-slide-skip.jpg" width="360" alt="Slide Skip"> |
| **Furniture Land** | 65% | <img src="media/parts/05-furniture-land.jpg" width="360" alt="Furniture Land"> |
| **Orange Hell** | 0% | <img src="media/parts/06-orange-hell.jpg" width="360" alt="Orange Hell"> |
| **Bell Tower** | 25% | <img src="media/parts/07-bell-tower.jpg" width="360" alt="Bell Tower"> |
| **The Hat** | 25% | <img src="media/parts/08-the-hat.jpg" width="360" alt="The Hat"> |
| **The Anvil** | 0% | <img src="media/parts/09-the-anvil.jpg" width="360" alt="The Anvil"> |
| **Floating Platforms** | 0% | <img src="media/parts/10-floating-platforms.jpg" width="360" alt="Floating Platforms"> |
| **The Bucket** | 0% | <img src="media/parts/11-the-bucket.jpg" width="360" alt="The Bucket"> |
| **Mountain Climb** | 0% | <img src="media/parts/12-mountain-climb.jpg" width="360" alt="Mountain Climb"> |
| **Satellite Climb** | 0% | <img src="media/parts/13-satellite-climb.jpg" width="360" alt="Satellite Climb"> |
| **Space** | 45% | <img src="media/parts/14-space.jpg" width="360" alt="Space"> |

## How it works

- A C# plugin (BepInEx) runs inside the game and connects it to Python: physics advances in lockstep with the agent,
  and the whole game can be saved and restored exactly.
- PPO networks (Stable-Baselines3) train on 10 copies of the game at once, about 6,000 steps per second on one PC.
- A search with save/load found climbs through every part. Their states start the training of 27 networks, each
  specialized in one stretch of the route and chained along it.
- Continuous games from the bottom show where the networks get stuck, and those spots get targeted training.

## Models

The networks and a video are on [Hugging Face](https://huggingface.co/miiwater/goi-rl).

# Getting Over It, played by reinforcement learning

<p align="center">
  <img src="media/climb.gif" width="480" alt="The networks climbing through the boxes and the furniture">
</p>

Neural networks playing the real *Getting Over It with Bennett Foddy*, with one goal: be the first to finish it with
reinforcement learning, in one continuous climb from the bottom to the top with no save/load.

**Best run so far: 63% of the way to the top**, in one continuous game from the bottom.

## Progress by part

The climb split into the parts speedrunners use. The project's own cuts are finer: 28 networks, each trained on its
own stretch of the route, so one part can hold several (the small print: the part's route progress, 0 at the bottom
and 864 at the top, and the project's stretches inside it). Two measures, **Solved** from 90%:

- **From the bottom**: of 64 continuous games from the bottom (an hour each, falls and re-climbs included), how many
  of those that reached the part got through it. A dash: no game has reached it yet.
- **From its entrance**: the networks play the part alone from one fixed state where a search's winning climb enters
  it, 20 tries of up to 10 minutes. Stricter: a single starting spot, and Orange Hell's falls every time.

The circle in the top corner of each picture is the networks' mouse input.

| Part | From the bottom | From its entrance | |
|---|---|---|---|
| **Coffee Rock**<br><sub>9–176: lakeside, rocks, tree and slope (2 networks)</sub> | Solved<br><sub>64 of 64</sub> | Solved | <img src="media/parts/01-coffee-rock.jpg" width="320" alt="Coffee Rock"> |
| **The Devil's Chimney**<br><sub>176–204: chimney (1 network)</sub> | Solved<br><sub>64 of 64</sub> | 85% | <img src="media/parts/02-the-devil-s-chimney.jpg" width="320" alt="The Devil's Chimney"> |
| **Construction Wall**<br><sub>204–245: Construction Wall, girder (3 networks)</sub> | Solved<br><sub>64 of 64</sub> | 65% | <img src="media/parts/03-construction-wall.jpg" width="320" alt="Construction Wall"> |
| **Slide Skip**<br><sub>245–310: hut, garden, playground (3 networks)</sub> | Solved<br><sub>63 of 64</sub> | 80% | <img src="media/parts/04-slide-skip.jpg" width="320" alt="Slide Skip"> |
| **Furniture Land**<br><sub>310–385: boxes, furniture (2 networks)</sub> | Solved<br><sub>63 of 63</sub> | 65% | <img src="media/parts/05-furniture-land.jpg" width="320" alt="Furniture Land"> |
| **Orange Hell**<br><sub>385–440: orange stretch, chapel foot, chapel jump (3 networks)</sub> | 68%<br><sub>43 of 63</sub> | 0% | <img src="media/parts/06-orange-hell.jpg" width="320" alt="Orange Hell"> |
| **Bell Tower**<br><sub>440–470: chapel top (3 networks)</sub> | 53%<br><sub>23 of 43</sub> | 25% | <img src="media/parts/07-bell-tower.jpg" width="320" alt="Bell Tower"> |
| **The Hat**<br><sub>470–530: roofs (1 network)</sub> | 35%<br><sub>8 of 23</sub> | 25% | <img src="media/parts/08-the-hat.jpg" width="320" alt="The Hat"> |
| **The Anvil**<br><sub>530–560: anvil boulder, anvil jump, cliff (3 networks)</sub> | 0%<br><sub>0 of 8</sub> | 0% | <img src="media/parts/09-the-anvil.jpg" width="320" alt="The Anvil"> |
| **Floating Platforms**<br><sub>560–628: snowballs (3 networks)</sub> | – | 0% | <img src="media/parts/10-floating-platforms.jpg" width="320" alt="Floating Platforms"> |
| **The Bucket**<br><sub>628–660: bucket (5 networks)</sub> | – | 0% | <img src="media/parts/11-the-bucket.jpg" width="320" alt="The Bucket"> |
| **Mountain Climb**<br><sub>660–726: Ice Mountain (2 networks)</sub> | – | 0% | <img src="media/parts/12-mountain-climb.jpg" width="320" alt="Mountain Climb"> |
| **Satellite Climb**<br><sub>726–760: ice platform, radio mast (3 networks)</sub> | – | 0% | <img src="media/parts/13-satellite-climb.jpg" width="320" alt="Satellite Climb"> |
| **Space**<br><sub>760–864: space (1 network)</sub> | – | 45% | <img src="media/parts/14-space.jpg" width="320" alt="Space"> |

## How it works

- A C# plugin (BepInEx) runs inside the game and connects it to Python: physics advances in lockstep with the agent,
  and the whole game can be saved and restored exactly.
- PPO networks (Stable-Baselines3) train on 10 copies of the game at once, about 6,000 steps per second on one PC.
- A search with save/load found climbs through every part. Their states start the training of 28 networks, each
  specialized in one stretch of the route and chained along it.
- Continuous games from the bottom show where the networks get stuck, and those spots get targeted training.

## Models

The networks and a video are on [Hugging Face](https://huggingface.co/miiwater/goi-rl).

## License

MIT. See [LICENSE](LICENSE).

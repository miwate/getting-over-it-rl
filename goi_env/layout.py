"""The bridge's observation layout and the client's fixes to the route progress it reads. Plain Python (no Windows
modules), so the JAX simulator (branch jax) shares it with goi_env.client."""

# Observation layout. The plugin appends n_rays distances from the pot to the terrain, counter-clockwise from +x.
OBS_FIELDS = (
    "pot_x", "pot_y", "pot_vx", "pot_vy", "pot_angle", "pot_angvel",  # world units, radians
    "hammer_cos", "hammer_sin", "hammer_angvel", "hammer_ext", "hammer_extvel",
    "head_contact", "pot_contact", "fell_in_lake", "won",  # 0/1 flags
    "progress",  # distance along the level route to the point nearest the pot (the game's ProgressMeter)
    "cursor_dx", "cursor_dy", "cursor_head_dx", "cursor_head_dy",  # the cursor the hammer aims at, from the pot and the head
    "mouse_x", "mouse_y", "mouse_speed",  # the controller's smoothed mouse input and running mouse speed
)
# The route ahead, relative to the pot: points this far further along it (Game.Ahead in the plugin).
AHEAD = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50)
OBS_FIELDS += tuple(f"ahead{d}_{axis}" for d in AHEAD for axis in ("dx", "dy"))


# Dead ends the level spline reads as headway, as (x0, x1, y0, y1, spline lo, hi, cap): progress there is capped. The
# trough between the anvil's boulder and the cliff (x ~70-73, y ~240-242; the anvil top is 537 at y ~243-245) reads
# 539.2: networks dropped in for +2.2 and sat there, never trying the jump from beside the anvil up to the cliff's ledge
# (545 at (73.8, 248.8)). ponytail: one hand-measured box; goi_env.route (progress along a climbed path) if more appear.
# The Snake's lip beside the bucket (x ~4-7, y ~262-265) reads 636-638, above the bucket itself (635) and the spot
# win-e4 jumped onto it from (~12, 266: 635): the approach network shuffled left onto the frictionless chute back to the
# start, or sat on its lip (film 10-26). Capped below the bucket. So is the overhang's face right of it (x ~8-13,
# y ~264-270): the spline climbs it (638 at (12.3, 270.2)) but nothing does; networks clung there for 636-639, more
# than the bucket paid, never reaching right for it, then slid onto the Snake (film 10-27).
DEAD_ENDS = ((69.5, 75.0, 230.0, 243.0, 535.0, 541.0, 537.0),
             (-5.0, 7.5, 254.0, 265.0, 633.0, 640.0, 634.5),
             (7.5, 14.0, 262.0, 273.0, 633.0, 641.0, 635.5))
# And ways the spline undervalues, as (x0, x1, y0, y1, spline lo, hi, floor): progress there is raised. The bucket
# (~(16.6, 270) at rest, on its rope) reads 635 like the spot the climber jumps onto it from: getting onto it paid
# nothing until a swing landed at 645 (approach starts 0.1-2.5% in training). Records count only while touching
# something, and in this box that is the bucket. 640: clearly above the capped approach (635.5).
FLOORS = ((14.0, 20.0, 267.0, 273.0, 633.0, 641.0, 640.0),)
_X, _Y, _P = (OBS_FIELDS.index(f) for f in ("pot_x", "pot_y", "progress"))


def dead_ends(obs):
    """obs with progress capped in DEAD_ENDS and raised in FLOORS."""
    for boxes, fix in ((DEAD_ENDS, min), (FLOORS, max)):
        for x0, x1, y0, y1, lo, hi, value in boxes:
            if x0 < obs[_X] < x1 and y0 < obs[_Y] < y1 and lo < obs[_P] < hi:
                return obs[:_P] + (fix(obs[_P], value),) + obs[_P + 1:]
    return obs

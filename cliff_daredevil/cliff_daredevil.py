import copy
from typing import Optional, Tuple

import Box2D as b2  # type: ignore
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .car_model import CAR_HEIGHT, CAR_WIDTH, CarModel

ROAD_HEIGHT = 23.0
DT = 1 / 60


class FrictionZoneListener(b2.b2ContactListener):
    def __init__(self, env):
        b2.b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        if (
            contact.fixtureA == self.env.friction_zone
            or contact.fixtureB == self.env.friction_zone
        ):
            if begin:
                self.env.friction_zone.touch = True
            else:
                self.env.friction_zone.touch = False


class CliffDaredevil(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        friction_profile: float = 0.1,
        friction_start: float = 40.0,
        safe_zone_reward: bool = True,
        old_gym_api: bool = False,
    ):
        self.contactListener_keepref = FrictionZoneListener(self)
        self.safe_zone_reward = safe_zone_reward
        self.old_gym_api = old_gym_api
        self.reward_fn = None
        if safe_zone_reward:
            self.reward_fn = self._returning_to_goal_reward
        self.min_position = -5.0
        self.max_position = 75.0
        self.goal_zone = (50.0, 50.0 + CAR_WIDTH)
        self.safe_zone = (10.0, 10.0 + CAR_WIDTH)
        self.cliff_edge = 0.3
        self.friction_start = friction_start
        self.friction = (
            friction_profile
            if callable(friction_profile)
            else lambda _: friction_profile
        )
        self.world = b2.b2World((0, -10), contactListener=self.contactListener_keepref)
        self._build_road()
        self.viewer = None
        self.action_space = spaces.Box(
            low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32
        )  # gas, brake
        self.observation_space = spaces.Box(
            low=np.array([self.min_position, -32.0]),
            high=np.array([self.max_position, 32.0]),
            dtype=np.float32,
        )  # position, velocity
        self.car: Optional[CarModel] = None
        self.render_mode = render_mode
        self.max_distance_to_goal_zone = self._set_max_distance_to_zone(self.goal_zone)
        self.max_distance_to_safe_zone = self._set_max_distance_to_zone(self.safe_zone)
        self._isopen = None
        self.seed()
        self.reset()

    def _returning_to_goal_reward(
        self, x_coord: float, goal: Tuple[float, float]
    ) -> float:
        reward = 0.0
        goal_mean = (goal[0] + goal[1]) / 2
        abs_distance_to_goal = np.abs(goal_mean - x_coord)
        reward = np.square(
            np.clip(
                (self.max_distance_to_safe_zone - abs_distance_to_goal)
                / self.max_distance_to_safe_zone,
                0.0,
                1.0,
            )
        )
        # if x_coord >= goal[0] and x_coord <= goal[1]:
        #     reward += 1.0
        return float(reward)

    def _set_max_distance_to_zone(self, zone: Tuple[float, float]) -> float:
        mean = (zone[0] + zone[1]) / 2
        max_distance_to_zone = float(
            max(
                np.abs(mean - (self.min_position + 3)),
                np.abs(mean - (self.max_position - 25.0)),
            )
        )
        return max_distance_to_zone

    def _build_road(self):
        self.ground = self.world.CreateStaticBody(
            fixtures=[
                b2.b2FixtureDef(
                    shape=b2.b2EdgeShape(
                        vertices=[
                            (self.min_position, ROAD_HEIGHT),
                            (self.goal_zone[1] + self.cliff_edge, ROAD_HEIGHT),
                        ]
                    ),
                    friction=0.99,
                ),
                b2.b2FixtureDef(
                    shape=b2.b2EdgeShape(
                        vertices=[
                            (self.friction_start, ROAD_HEIGHT),
                            (self.goal_zone[1], ROAD_HEIGHT),
                        ]
                    ),
                    friction=self.friction(0),
                ),
            ]
        )
        self.ground.fixtures[1].touch = False
        self.friction_zone = self.ground.fixtures[1]
        self.ground.userData = self.ground

    # In new Gym API, this function is deprecated
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        truncated = False
        if action is not None:
            action = np.clip(action, -1.0, 1.0)
            self.car.gas(float(action[0]))
            self.car.brake(float(action[1]))

        self.car.step()
        self.world.Step(DT, 6 * 30, 2 * 30)
        x, y = self.car.hull.position
        if self.friction_zone.touch:
            progress = np.clip(
                (x - self.friction_start) / (self.goal_zone[1] - self.friction_start),
                0.0,
                1.0,
            )
            friction = self.friction(progress)
            self.friction_zone.friction = friction

        angle = self.car.hull.angle
        backward = x < self.min_position
        terminated = bool(y < ROAD_HEIGHT or backward or np.abs(angle) > 0.9)

        # Compute reward
        if self.reward_fn is None:
            reward = 0
            if backward:
                reward -= 1.0
            if self.goal_zone[0] < x < self.goal_zone[1]:
                reward += 1.0
            distance = self.goal_zone[0] - x
            reward -= np.square(distance)
        elif self.safe_zone_reward:
            reward = self.reward_fn(x, self.safe_zone)

        # Compute cost
        cost = (
            -((self.goal_zone[0] + self.goal_zone[1]) / 2 + self.cliff_edge - x)
            / self.max_distance_to_goal_zone
        )

        # New state
        v = self.car.hull.linearVelocity[0]

        if self.render_mode == "human":
            self.render()

        if self.old_gym_api:
            return (
                np.array([x, v], np.float32),
                reward,
                terminated,
                {"cost": cost},
            )

        return (
            np.array([x, v], np.float32),
            reward,
            terminated,
            truncated,
            {"cost": cost},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self._destroy()
        self.friction_zone.touch = False
        self.friction_zone.friction = self.friction(0.0)  # type: ignore
        # Starting position after reset
        position = self.np_random.uniform(
            low=self.min_position + 3, high=self.max_position - 25.0
        )
        velocity = self.np_random.normal(loc=0.0, scale=(32 / 2))
        self.car = CarModel(self.world, position, ROAD_HEIGHT + 0.5, velocity)

        if self.render_mode == "human":
            self.render()

        if self.old_gym_api:
            return self.step(None)[0]
        return self.step(None)[0], {}

    def manual_reset(self, x_coord: float, vel: float):
        self.reset()
        self._destroy()
        self.friction_zone.touch = False
        self.friction_zone.friction = self.friction(0.0)  # type: ignore
        # Starting position after reset
        position = x_coord
        velocity = vel
        self.car = CarModel(self.world, position, ROAD_HEIGHT + 0.5, velocity)

        if self.render_mode == "human":
            self.render()

        if self.old_gym_api:
            return self.step(None)[0]
        return self.step(None)[0], {}

    def _destroy(self):
        if self.car is None:
            return
        self.car.destroy()

    def render(self):
        mode = self.render_mode
        screen_width, screen_height = 640, 320
        if self.viewer is None:
            import cliff_daredevil.rendering as rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(self.min_position, self.max_position, 0.0, 40.0)
            sky = rendering.make_polygon(
                [
                    (self.min_position, 0.0),
                    (self.min_position, 40.0),
                    (self.max_position, 40.0),
                    (self.max_position, 0.0),
                ]
            )
            sky.set_color(135 / 255, 206 / 255, 235 / 255)
            xs_ground = np.linspace(self.goal_zone[1], self.goal_zone[1] + 1.0, 25)
            ys_ground = np.linspace(ROAD_HEIGHT, 0.0, 25)
            xs_ground += self.np_random.uniform(-0.75, 0.75, 25)
            ground = rendering.make_polygon(
                [(self.min_position, 0.0), (self.min_position, ROAD_HEIGHT)]
                + [*zip(xs_ground, ys_ground)]
            )
            ground.set_color(237 / 255, 201 / 255, 175 / 255)
            oil = rendering.make_polyline(
                [
                    (self.friction_start, ROAD_HEIGHT - 0.1),
                    (self.goal_zone[1], ROAD_HEIGHT - 0.1),
                ]
            )
            oil.set_linewidth(2)
            xs_sea = np.linspace(self.goal_zone[1], self.max_position, 25)
            ys_sea = np.maximum(np.sin(xs_sea * 7.1) * 2.0, 0.2)
            sea = rendering.make_polygon(
                [*zip(xs_sea, ys_sea)] + [(xs_sea[-1], 0.0), (xs_sea[0], 0.0)]
            )
            sea.set_color(0, 105 / 255, 148 / 255)
            sun = rendering.make_circle(2.5)
            sun.set_color(252 / 255, 212 / 255, 64 / 255)
            sun.add_attr(rendering.Transform((65, 35)))
            car_width, car_height = CAR_WIDTH, CAR_HEIGHT
            l, r, t, b = -car_width / 2, car_width / 2, car_height, 0
            car = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
            self.car_transform = rendering.Transform()
            car.add_attr(self.car_transform)
            radius = car_height / 2.5
            frontwheel = rendering.make_circle(radius)
            frontwheel.add_attr(rendering.Transform(translation=(car_width / 4, 0)))
            frontwheel.add_attr(self.car_transform)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel_rim = rendering.make_circle(0.3, res=30, filled=True)
            frontwheel_rim.set_color(1.0, 0.0, 0.0)
            frontwheel_rim.add_attr(rendering.Transform(translation=(radius - 0.3, 0)))
            self.frontwheel_rim_transform = rendering.Transform()
            frontwheel_rim.add_attr(self.frontwheel_rim_transform)
            backwheel = rendering.make_circle(radius)
            backwheel.add_attr(rendering.Transform(translation=(-car_width / 4, 0)))
            backwheel.add_attr(self.car_transform)
            backwheel.set_color(0.5, 0.5, 0.5)
            backwheel_rim = rendering.make_circle(0.3, res=30, filled=True)
            backwheel_rim.set_color(1.0, 0.0, 0.0)
            backwheel_rim.add_attr(rendering.Transform(translation=(radius - 0.3, 0)))
            self.backwheel_rim_transform = rendering.Transform()
            backwheel_rim.add_attr(self.backwheel_rim_transform)

            def make_flag(position, color=(0.8, 0.8, 0)):
                flagx = position
                flagy1 = ROAD_HEIGHT
                flagy2 = flagy1 + 2.0
                flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
                flag = rendering.FilledPolygon(
                    [
                        (flagx, flagy2),
                        (flagx, flagy2 - 1.0),
                        (flagx + 2.5, flagy2 - 0.5),
                    ]
                )
                flag.set_color(*color)
                return flag, flagpole

            right_flag, right_flagpole = make_flag(self.goal_zone[0])
            left_flag, left_flagpole = make_flag(self.goal_zone[1])
            safe_right_flag, safe_right_flagpole = make_flag(
                self.safe_zone[0], color=(0.0, 0.0, 0.8)
            )
            safe_left_flag, safe_left_flagpole = make_flag(
                self.safe_zone[1], color=(0.0, 0.0, 0.8)
            )
            self.viewer.add_geom(sky)
            self.viewer.add_geom(sea)
            self.viewer.add_geom(ground)
            self.viewer.add_geom(oil)
            self.viewer.add_geom(sun)
            self.viewer.add_geom(car)
            self.viewer.add_geom(frontwheel)
            self.viewer.add_geom(frontwheel_rim)
            self.viewer.add_geom(backwheel)
            self.viewer.add_geom(backwheel_rim)
            # FLAGS SAFE ZONES
            self.viewer.add_geom(safe_right_flagpole)
            self.viewer.add_geom(safe_right_flag)
            self.viewer.add_geom(safe_left_flagpole)
            self.viewer.add_geom(safe_left_flag)
            # FLAGS GOAL ZONES
            self.viewer.add_geom(right_flagpole)
            self.viewer.add_geom(right_flag)
            self.viewer.add_geom(left_flagpole)
            self.viewer.add_geom(left_flag)
        pos = self.car.hull.position
        self.car_transform.set_translation(*pos)
        self.car_transform.set_rotation(self.car.hull.angle)
        self.frontwheel_rim_transform.set_translation(*pos)
        self.backwheel_rim_transform.set_translation(*pos)
        self.frontwheel_rim_transform.set_translation(*self.car.wheels[0].position)
        self.frontwheel_rim_transform.set_rotation(self.car.wheels[0].angle)
        self.backwheel_rim_transform.set_rotation(self.car.wheels[1].angle)
        self.backwheel_rim_transform.set_translation(*self.car.wheels[1].position)

        self._isopen, frame = self.viewer.render(return_rgb_array=(mode == "rgb_array"))

        return frame

    @property
    def isopen(self):
        if self._isopen is None:
            print("Warning: isopen not defined for this viewer. Returning False.")
            return False
        return self._isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    from gym.wrappers import TimeLimit
    from pyglet.window import key  # type: ignore

    a = np.array([0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.SPACE:
            a[1] = +0.8

    def key_release(k, mod):
        if k == key.RIGHT:
            a[0] = 0
        if k == key.LEFT:
            a[0] = 0
        if k == key.SPACE:
            a[1] = 0

    env: gym.Env = CliffDaredevil(friction_profile=0.01, friction_start=25.0)
    env = TimeLimit(env, 1000)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    isopen = True
    while isopen:
        env.reset()
        env.manual_reset(env.max_position - 25, 0)
        total_reward = 0.0
        total_cost = 0.0
        steps = 0
        restart = False
        while True:
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            total_cost += info["cost"]
            if steps % 1 == 0 or terminated:
                print("\naction " + str(["{:+0.4f}".format(x) for x in a]))
                print(
                    "step {} observation {} type {}".format(
                        env._elapsed_steps, s, type(s)
                    )
                )
                print("step {} total_reward {:+0.4f}".format(steps, total_reward))
                print("step {} total_cost {:+0.4f}".format(steps, total_cost))
                print(
                    "step {} cost {:+0.4f} type {}".format(
                        steps, info["cost"], type(info["cost"])
                    )
                )
                print("step {} reward {:+0.4f} type {}".format(steps, r, type(r)))
                print(
                    "step {} terminal {} type {}".format(
                        steps, terminated, type(terminated)
                    )
                )
                print(
                    "step {} truncated {} type {}".format(
                        steps, truncated, type(truncated)
                    )
                )
            steps += 1
            _ = env.render()  # type: ignore
            isopen = copy.copy(env.isopen)

            if terminated or truncated or restart or not isopen:
                break
    env.close()

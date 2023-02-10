import importlib

import pytest

from cliff_daredevil import CliffDaredevil

gymnasium_loader = importlib.find_loader("gymnasium")


@pytest.mark.skipif(
    gymnasium_loader is None,
    reason="Running old version of gym",
)
def test_env():
    import gymnasium as gym  # noqa: F401

    old_gym = False
    ##
    # check safe_zone version
    env = CliffDaredevil(safe_zone_reward=True, old_gym_api=old_gym)
    env.reset()
    gym.utils.env_checker.check_env(env=env, warn=True, skip_render_check=False)
    # check normal reward version
    env = CliffDaredevil(safe_zone_reward=False, old_gym_api=old_gym)
    env.reset()
    gym.utils.env_checker.check_env(env=env, warn=True, skip_render_check=False)


@pytest.mark.skipif(gymnasium_loader is not None, reason="Running new version of gym")
def test_old_env():
    try:
        import gym  # noqa: F401
    except ImportError:
        print("old gym not installed")

    try:
        from stable_baselines3.common import env_checker  # noqa: F401
    except ImportError:
        print("stable_baselines3 not installed")

    old_gym = True
    ##
    # check safe_zone version
    env = CliffDaredevil(safe_zone_reward=True, old_gym_api=old_gym)
    env.reset()
    env_checker.check_env(env=env, warn=True, skip_render_check=False)
    # check normal reward version
    env = CliffDaredevil(safe_zone_reward=False, old_gym_api=old_gym)
    env.reset()
    env_checker.check_env(env=env, warn=True, skip_render_check=False)

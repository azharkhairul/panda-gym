import os

from gym.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    register(
        id="PandaReach{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="PandaPush{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaPushEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="PandaSlide{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaSlideEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="PandaPickAndPlaceDenser{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaPickAndPlaceDenseEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="PandaPickAndPlace{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaPickAndPlaceEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="PandaPickAndPlaceBarrier{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaPickAndPlaceBarrierEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="PandaPickAndPlaceCluttered{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaPickAndPlaceClutteredEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="PandaPickAndPlaceClutteredMoreObj{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaPickAndPlaceClutteredMoreObjEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id="PandaPickAndPlaceConvergence{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaPickAndPlaceConvergenceEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="PandaStack{}-v1".format(suffix),
        entry_point="panda_gym.envs:PandaStackEnv",
        kwargs=kwargs,
        max_episode_steps=100,
    )

import os
import gym
import sys
from gym_idsgame.agents.manual_agents.manual_defense_agent import ManualDefenseAgent

def get_script_path():
    """
    :return: the script path
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def default_output_dir() -> str:
    """
    :return: the default output dir
    """
    script_dir = get_script_path()
    return script_dir

# Program entrypoint
if __name__ == '__main__':
    # Keybinds: Change attack/defense type by clicking numbers 1-10, after switching to the right type, click on a server to attack/defend,
    # click SPACE to reset the game. If you
    versions = range(0, 20)
    random_seed = 0
    version = versions[3]
    #env_name = "idsgame-random_attack-v" + str(version) # Play as Defender
    env_name = "idsgame-maximal_attack-v" + str(version) # Play as Defender
    # env_name = "idsgame-minimal_defense-v" + str(version) # Play as Attacker
    # env_name = "idsgame-random_defense-v" + str(version) # Play as Attacker
    env = gym.make(env_name)
    ManualDefenseAgent(env.idsgame_config)
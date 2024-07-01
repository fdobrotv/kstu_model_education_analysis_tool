from gymnasium import envs
from gymnasium.envs.registration import EnvSpec

def get_atari_games():
    all_envs = envs.registry

    ale = 0
    aleIds = set()
    aleRawNames = set()
    aleNames = set()
    shimmy = 0
    shimmyIds = set()
    shimmyRawNames = set()
    shimmyDeterministicNames = set()
    shimmyNoFrameskipNames = set()
    shimmyNames = set()
    other = 0
    otherIds = set()
    otherRawNames = set()
    for key in all_envs:
        value: EnvSpec = all_envs[key]
        if value.namespace == 'ALE':
            ale += 1
            aleIds.add(value.id)
            aleRawNames.add(value.name)
            splittedName = value.name.split("-", 2)
            aleNames.add(splittedName[0])
        elif value.entry_point == 'shimmy.atari_env:AtariEnv':
            shimmy += 1
            shimmyIds.add(value.id)
            shimmyRawNames.add(value.name)
            splittedName = value.name.split("-", 2)
            
            game_name = splittedName[0]
            if ("Deterministic" in game_name):
                shimmyDeterministicNames.add(game_name)
            elif ("NoFrameskip" in game_name):
                shimmyNoFrameskipNames.add(game_name)
            else:
                shimmyNames.add(game_name)
        else:
            other += 1
            otherIds.add(value.id)
            otherRawNames.add(value.name)
            # print(key, "->", value)

    aleNames = sorted(aleNames)
    shimmyNames = sorted(shimmyNames)
    otherRawNames = sorted(otherRawNames)
    print(f"Counts for ale: {ale}, shimmy: {shimmy}")
    return aleNames, shimmyNames, otherRawNames
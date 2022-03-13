from marlgrid.envs import env_from_config
from ddrqn_agent import DDRQN_Agent
def main(**argv):
    
    env_config =  {
        "env_class": "ClutteredGoalCycleEnv",
        "grid_size": 13,
        "max_steps": 250,
        "clutter_density": 0.15,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": False,
        "n_bonus_tiles": 3,
        "initial_reward": True,
        "penalty": -1.5
    }

    agent_interface_config = {
        'view_tile_size': 3,
        'view_size': 7,
        'view_offset': 3,
        'observation_style': 'rich',
        'prestige_beta': 0.95, # determines the rate at which prestige decays
        'color': 'prestige',
        'spawn_delay': 100,
    }

    ddqrn_learning_config = {
        "batch_size": 8,
        'num_minibatches': 10,
        "minibatch_size": 256,
        "minibatch_seq_len": 8,
        "hidden_update_interval": 10,

        'learning_rate': 1.e-4, # 1.e-3, #
        "kl_target":  0.01,
        "clamp_ratio": 0.2,
        "lambda":0.97,
        "gamma": 0.99,
        'entropy_bonus_coef': 0.0,#0001,
        'value_loss_coef': 1.0,
    }

    n_new_agents = 4 # Number of new agents to be created with the above config/hyperparameters.
    grid_agents = []
    new_agents_info = [
        {'interface_config': agent_interface_config, 'learning_config': ddqrn_learning_config}
        for _ in range(n_new_agents)
    ]

    env = env_from_config(env_config)

    for agent_info in new_agents_info:
        iface = GridAgentInterface(**agent_info['interface_config'])
        new_fella = DDRQN_Agent(
            observation_space=iface.observation_space,
            action_space=iface.action_space, 
            learning_config=agent_info['learning_config'],
            model_config=agent_info['model_config'],
        )
        new_fella.metadata['marlgrid_interface'] = agent_interface_config
        grid_agents.append(new_fella)

    agents = IndependentAgents(*grid_agents)

    agents.set_device(device)
    print(f"Agents have {count_parameters(agents.agents[0].ac)} parameters.")



if __name__ == "__main__":
    main()
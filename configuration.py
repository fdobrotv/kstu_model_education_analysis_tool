from atari_env_helper import get_atari_games
import time

aleNames, shimmyNames, otherRawNames = get_atari_games()

class DataValues():
    #Going to be initialized late, on start process
    tensorboard_url = None
    tensorboard_thread = None
    ctx = None
    child = None
    data_queue = None
    manager = None
    shared_params = None
    start_event = None
    stop_event = None
    global_stop_event = None
    progress_bar_stop_event = None
    statistics_stop_event = None
    experiment_finished_event = None
    log_queue = None
    ui_log_queue = None
    iteration_count = None
    
    procs = []
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    algorithm = "r2d2"
    image_source = "Shimmy"
    environment_names = shimmyNames
    
    environment_name = "Pong"
    environment_height = 84
    environment_width = 84
    environment_frame_skip = 4
    environment_frame_stack = 1
    compress_state = True
    num_actors = 2 #multiprocessing.cpu_count()
    replay_capacity = 100000
    min_replay_size = 1000
    clip_grad = True
    max_grad_norm = 40.0
    
    learning_rate = 0.0001
    adam_eps = 0.0001
    discount = 0.997
    
    unroll_length = 80
    burn_in = 40
    
    batch_size = 32
    priority_exponent = 0.9
    importance_sampling_exponent = 0.6
    normalize_weights = True
    gpu_actors = True
    
    priority_eta = 0.9
    rescale_epsilon = 0.001
    n_step = 5
    
    num_iterations = 100
    num_train_steps = int(2e5) # 5e5 originally, but need to complete the task in one day on 4090+7950
    
    num_eval_steps = int(2e4)
    max_episode_steps = 108000
    target_net_update_interval = 1500
    actor_update_interval = 400
    eval_exploration_epsilon = 0.01
    
    seed = 0
    
    use_tensorboard = True
    actors_on_gpu = True
    
    debug_screenshots_interval = 0
    
    tag = f"S{seed}"
    
    results_csv_path = "./logs/r2d2_atari_results.csv"
    checkpoint_dir = "./checkpoints"
    
    def update_tag():
        myDict = {
            "": DataValues.timestr, 
            "S": DataValues.seed,
            "LR": f"{DataValues.learning_rate:.4f}",
            "ADAM": f"{DataValues.adam_eps:.4f}",
            "DIS": f"{DataValues.discount:.4f}",
            "BS": f"{DataValues.batch_size}",
        }
        mySeparator = "-"

        x = mySeparator.join(str(f"{x[0]}{x[1]}") for x in myDict.items())
        DataValues.tag = x
        # DataValues.tag = f"{DataValues.timestr}-S{DataValues.seed}-LR{DataValues.learning_rate:.4f}-ADAM{DataValues.adam_eps:.4f}-DIS{DataValues.discount:.4f}-BS{DataValues.batch_size}"
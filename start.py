import dearpygui.dearpygui as dpg
import time
from absl import app
import multiprocessing
import sys
from atari_env_helper import get_atari_games

aleNames, shimmyNames, otherRawNames = get_atari_games()

class DataValues():
    # stop_event = multiprocessing.Event()
    procs = []
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    algorithm = "r2d2"
    image_source = "Shimmy"
    # environment_names = shimmyNames
    
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
    num_train_steps = 5e5
    
    num_eval_steps = 2e4
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

def start_experiment():
    
    DataValues.update_tag()
    
    # python3 -m deep_rl_zoo.r2d2.run_atari 
    # --environment_name=Breakout 
    # seed=1 
    # --learning_rate=0.0001 
    # --discount=0.997 
    # --replay_capacity=100000 
    # --batch_size=64  
    # --tag=14.04.24-08:34-S1-LR0.0001-DIS0.997-RC100000-BS64 
    # --actors_on_gpu
    # p = subprocess.Popen(['python3', '-m', f"deep_rl_zoo.{DataValues.algorithm}.run_atari", 
    #                         '--environment_name', DataValues.environment_name, 
    #                         '--seed', DataValues.seed, 
    #                         '--learning_rate', DataValues.learning_rate, 
    #                         '--discount', DataValues.discount,
    #                         '--replay_capacity', DataValues.replay_capacity,
    #                         '--batch_size', DataValues.batch_size,
    #                         '--tag', DataValues.tag,
    #                         '--actors_on_gpu'
    # ])
    # from deep_rl_zoo.{DataValues.algorithm}.run_atari import main as selected_main
    import importlib, os, pkgutil
    # from . import deep_rl_zoo
    # sys.path.append(f"deep_rl_zoo.{DataValues.algorithm}")
    # from run_atari import __main__ 
    
    # [name for _, name, _ in pkgutil.iter_modules(['testpkg'])]
    # pkgpath = os.path.dirname(deep_rl_zoo.__file__)
    print (DataValues.algorithm)
    package = f"./deep_rl_zoo/{DataValues.algorithm}"
    sys.path.append(package)
    print (f"Package is: {package}")
    
    import run_atari
    # from absl.flags import FLAGS
    # print([name for _, name, _ in pkgutil.iter_modules([package])])
    
    # from os.path import dirname, basename, isfile, join
    # import glob
    # modules = glob.glob(join(dirname(package), "*.py"))
    # __entrypoint__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
    # print(__all__)
    
    # selected_main = importlib.import_module(f"run_atari", package=package)
    # print(selected_main)
    
    
    
    # # __all__.
    
    # import imp
    # file, pathname, description = imp.find_module('__main__', [package])
    
    # file, pathname, description = imp.find_module('run_atari', [f"deep_rl_zoo.{DataValues.algorithm}"])
    # my_module = imp.load_module('run_atari', file, pathname, description)
    
    # # selected_main = importlib.import_module(f"..run_atari", package=f"deep_rl_zoo.{DataValues.algorithm}")
    
    # print(f"DataValues.tag is: {DataValues.tag}")
    
    # FLAGS.environment_name
    
    
    # multiprocessing.set_start_method('spawn')
    
    
    # deep_rl_zoo.r2d2.run_atari 
    # --environment_name=BeamRider 
    # --seed=1 
    # --learning_rate=0.0001 
    # --discount=0.997 
    # --replay_capacity=100000 
    # --batch_size=64  
    # --tag=19.04.24-10:41-S1-LR0.0001-DIS0.997-RC100000-BS64 
    # --actors_on_gpu
    arguments = ["main", 
                 "--environment_name", f"{DataValues.environment_name}",
                 "--seed", f"{DataValues.seed}",
                 "--learning_rate", f"{DataValues.learning_rate}",
                 "--discount", f"{DataValues.discount}",
                 "--replay_capacity", f"{DataValues.replay_capacity}",
                 "--batch_size", f"{DataValues.batch_size}",
                 "--num_actors", f"{DataValues.num_actors}",
                 "--tag", f"{DataValues.tag}",
                 "--actors_on_gpu"
                 ]
    print(arguments)
    
    # import time
    # def progressAsync(sender, data):
    #     print(data)
    #     dpg.set_value("progress", value=0)
    #     time.sleep(data)
    #     counter = 0.0
    #     max_time = float(data)
    #     while counter <= 1:
    #         dpg.set_value("progress", value=counter)
    #         counter += 0.1
    #         print(counter)
    
    
    
    # waitTime = 5
    # dpg.run_async_function("progressAsync", waitTime)
    
    import absl
    from absl import flags
    # flags.set_default(FLAGS.environment_name, DataValues.environment_name)
    
    
    # FLAGS.append_flags_into_file() on the parent process side and FLAGS.read_flags_from_files('--flagfile=path/to/said/file') + .mark_as_parsed() in the child process before using FLAGS.
    
    app.run(run_atari.main, argv=arguments)
    # import multiprocessing
    # process = multiprocessing.Process(target=app.run, args=(run_atari.main, arguments))
    # DataValues.procs.append(process)
    # process.start()
    
data = DataValues()

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=600, height=300)

with dpg.window(label="Learn start window", width=850, height=900):
    with dpg.tab_bar(tag="main_tab_bar") as tb:
        with dpg.tab(label="learning", tag="learning_tab"):
            dpg.add_button(label="Start!", callback=start_experiment)
        

if __name__ == "__main__":
    start_experiment()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    
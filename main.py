import multiprocessing.context
import multiprocessing.context
import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import numpy as np
import shlex, subprocess
from absl import app
# from absl import flags
# from absl import logging
import multiprocessing
from ctypes import c_int, c_char_p
from multiprocessing import Process
import sys
import logging
import threading
import time

import configuration
from configuration import DataValues
from tensorboard import program
from os.path import exists
import queue

def normalize_between_0_and_1(value):
    # New value = (value – min) / (max – min)
    # return (value - 0) / ((DataValues.num_iterations * DataValues.num_train_steps) - 0)
    return value / DataValues.num_iterations
    
def update_tag(sender, value, app_data):
	dpg.set_value("tag", DataValues.tag)
 
def update_algorithm(sender, value, app_data):
    DataValues.algorithm = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_image_source(sender, value, app_data):
    DataValues.image_source = value
    DataValues.update_tag()
    
    if DataValues.image_source == "ALE":
        DataValues.environment_names = configuration.aleNames
    elif DataValues.image_source == "Shimmy":
        DataValues.environment_names = configuration.shimmyNames
    else:
        DataValues.environment_names = configuration.otherRawNames
    dpg.configure_item("EnvNameSelector", items=DataValues.environment_names)
    
    update_tag(sender, value, app_data)
    
def update_environment_name(sender, value, app_data):
    DataValues.environment_name = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_seed(sender, value, app_data):
    DataValues.seed = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_environment_frame_skip(sender, value, app_data):
    DataValues.environment_frame_skip = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_environment_frame_stack(sender, value, app_data):
    DataValues.environment_frame_stack = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_compress_state(sender, value, app_data):
    DataValues.compress_state = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_num_actors(sender, value, app_data):
    DataValues.num_actors = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_replay_capacity(sender, value, app_data):
    DataValues.replay_capacity = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_min_replay_size(sender, value, app_data):
    DataValues.min_replay_size = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_clip_grad(sender, value, app_data):
    DataValues.clip_grad = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_max_grad_norm(sender, value, app_data):
    DataValues.max_grad_norm = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_learning_rate(sender, value, app_data):
    DataValues.learning_rate = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_adam_eps(sender, value, app_data):
    DataValues.adam_eps = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_discount(sender, value, app_data):
    DataValues.discount = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_unroll_length(sender, value, app_data):
    DataValues.unroll_length = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_burn_in(sender, value, app_data):
    DataValues.burn_in = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_batch_size(sender, value, app_data):
    DataValues.batch_size = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_priority_exponent(sender, value, app_data):
    DataValues.priority_exponent = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_importance_sampling_exponent(sender, value, app_data):
    DataValues.importance_sampling_exponent = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
 
def update_normalize_weights(sender, value, app_data):
    DataValues.normalize_weights = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_gpu_actors(sender, value, app_data):
    DataValues.gpu_actors = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)   
 
def update_priority_eta(sender, value, app_data):
    DataValues.priority_eta = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_rescale_epsilon(sender, value, app_data):
    DataValues.rescale_epsilon = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_n_step(sender, value, app_data):
    DataValues.n_step = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_num_iterations(sender, value, app_data):
    DataValues.num_iterations = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_num_train_steps(sender, value, app_data):
    DataValues.num_train_steps = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_num_eval_steps(sender, value, app_data):
    DataValues.num_eval_steps = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_max_episode_steps(sender, value, app_data):
    DataValues.max_episode_steps = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_target_net_update_interval(sender, value, app_data):
    DataValues.target_net_update_interval = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_actor_update_interval(sender, value, app_data):
    DataValues.actor_update_interval = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_eval_exploration_epsilon(sender, value, app_data):
    DataValues.eval_exploration_epsilon = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def update_use_tensorboard(sender, value, app_data):
    DataValues.use_tensorboard = value
    DataValues.update_tag()
    
    update_tag(sender, value, app_data)
    
def run_experiment_start_task(
    algorithm,
    arguments: dict,
    ctx: multiprocessing.context.SpawnContext,
    data_queue,
    shared_params,
    start_event,
    stop_event,
    global_stop_event,
    log_queue,
    ui_log_queue,
    iteration_count,
    procs,
    experiment_finished_event,
):
    logging.debug(f"Selected algorithm: {algorithm}")
    package = f"./deep_rl_zoo/{algorithm}"
    sys.path.append(package)
    logging.debug(f"Package is: {package}")
    
    import run_atari_plain
    
    logging.debug(f"Arguments: {arguments}")
    
    learning_process = ctx.Process(name="train_job", target=run_atari_plain.main, args=(
        arguments,
        data_queue,
        shared_params,
        start_event,
        stop_event,
        global_stop_event,
        log_queue,
        ui_log_queue,
        iteration_count,
        experiment_finished_event,
    ))
    
    # AssertionError: daemonic processes are not allowed to have children
    # learning_process.daemon = True
    
    # TODO: Look how to solve this control via multiprocessing context
    procs.append(learning_process)
    learning_process.start()
    logging.debug("Main: before join()")
    learning_process.join()       
    
    logging.debug(f"Experiment start task stopped!")
    
    
def run_progress_bar_task(
    progress_bar_stop_event: multiprocessing.Event,
    iteration_count: multiprocessing.Value
):
    while True:
        try:
            time.sleep(10)
            logging.debug(f"Update progress bar iteration: {iteration_count.value}")
            if progress_bar_stop_event.is_set():
                logging.debug("Update progress bar, break by progress_bar_stop_event")
                break
            current_iteration = iteration_count.value
            total_position = current_iteration
            normalized = normalize_between_0_and_1(total_position)
            logging.debug(f"Update progress bar, normalized: {normalized}")
            dpg.set_value("progress_bar", value=normalized)
            continue
        except Exception as e:
            logging.error(f"Update progress EXCEPTION! {e}")
    
    progress_bar_stop_event.clear()        
    logging.debug(f"Update progress bar stopped!")

def run_statistics_task(
    statistics_stop_event: multiprocessing.Event,
    ui_log_queue: multiprocessing.Queue
):
    episonde_returns = []
    step_rates = []
    durations = []
    while True:
        try:
            time.sleep(10)
            logging.debug(f"Update statistics")
            ui_log_output = ui_log_queue.get()
            
            current_iteration = ui_log_output[0][1]
            # current_iteration = ui_log_output["current_iteration"]
            dpg.set_value("current_iteration", value=current_iteration)
            current_step = ui_log_output[2][1]
            dpg.set_value("current_step", value=current_step)
            
            episode_return = ui_log_output[3][1]
            episonde_returns.append(episode_return)
            step_rate = ui_log_output[5][1]
            step_rates.append(step_rate)
            duration = ui_log_output[6][1]
            durations.append(duration)
            
            dpg.set_value("max_episode_return", value=max(episonde_returns))
            dpg.set_value("max_step_rate", value=max(step_rates))
            dpg.set_value("max_duration", value=max(durations))
            
            dpg.set_value("min_episode_return", value=min(episonde_returns))
            dpg.set_value("min_step_rate", value=min(step_rates))
            dpg.set_value("min_duration", value=min(durations))
            
            if statistics_stop_event.is_set():
                logging.debug("Update statistics, break by statistics_stop_event")
                break
            
            continue
        except queue.Empty:
            pass
        except EOFError:
            pass
        except Exception as e:
            logging.error(f"Update statistics EXCEPTION! {e}")
    
    statistics_stop_event.clear()        
    logging.debug(f"Update statistics stopped!")
    
def run_experiment_finished_task(
    experiment_finished_event: multiprocessing.Event
):
    while True:
        try:
            time.sleep(10)
            if experiment_finished_event.is_set():
                logging.debug("Experiment_finished, break by experiment_finished_event")
                handle_experiment_finished()
                break
            
            continue
        except queue.Empty:
            pass
        except EOFError:
            pass
        except Exception as e:
            logging.error(f"Update experiment_finished EXCEPTION! {e}")
    
    experiment_finished_event.clear()        
    logging.debug(f"Update experiment_finished stopped!")
    
def run_tensorboard_task(
    tensorboard_url: multiprocessing.Value,
    tensorboard_thread: multiprocessing.Value
):
    # logging.getLogger("tensorflow").setLevel(logging.ERROR)
    # logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
    tracking_address = "runs" # the path of your log file.

    tb = program.TensorBoard()
    tb.configure(argv=(None, '--logdir', tracking_address))
    url = tb.launch()
    # url += "?runFilter=163922"
    tensorboard_url = url
    logging.debug(f"Tensorflow listening on {tensorboard_url}")
    
    time.sleep(1800)
    tensorboard_thread = None

def stopped_state():
    dpg.disable_item("stop_button")
    dpg.disable_item("tensorboard_button_v2")
    dpg.hide_item("stop_button")
    dpg.hide_item("tensorboard_button_v2")
    dpg.enable_item("start_button")
    dpg.show_item("start_button")
    dpg.hide_item("current_statistics")
    # dpg.hide_item("max_statistics")
    # dpg.hide_item("min_statistics")
    
def stop_experiment():
    print ("stop_experiment() called")
    stopped_state()
    dpg.hide_item("progress_bar")
    
    DataValues.global_stop_event.set()
    DataValues.progress_bar_stop_event.set()
    DataValues.statistics_stop_event.set()
    
    for proc in DataValues.procs:
        print('Going to terminate child proc pid: {}'.format(proc))
        proc.join()
        proc.close()
        print('Terminated child proc pid: {}'.format(proc))
        
    import psutil

    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        # child.terminate()
        print('Going to terminate child pid: {}'.format(child.pid))
        # time.sleep(0.1)
        child.join()
        child.close()
        print('Terminated child pid: {}'.format(child.pid))
    # DataValues.manager.close()
        
        
def run_experiment_start_separately():
    
    arguments = ["main", 
                 "--environment_name", f"{DataValues.environment_name}",
                 "--seed", f"{DataValues.seed}",
                 "--learning_rate", f"{DataValues.learning_rate}",
                 "--adam_eps", f"{DataValues.adam_eps}",
                 "--discount", f"{DataValues.discount}",
                 "--replay_capacity", f"{DataValues.replay_capacity}",
                 "--min_replay_size", f"{DataValues.min_replay_size}",
                 "--clip_grad", f"{DataValues.clip_grad}",
                 "--max_grad_norm", f"{DataValues.max_grad_norm}",
                 "--batch_size", f"{DataValues.batch_size}",
                 "--priority_exponent", f"{DataValues.priority_exponent}",
                 "--importance_sampling_exponent", f"{DataValues.importance_sampling_exponent}",
                 "--normalize_weights", f"{DataValues.normalize_weights}",
                 "--gpu_actors", f"{DataValues.gpu_actors}",
                 "--priority_eta", f"{DataValues.priority_eta}",
                 "--rescale_epsilon", f"{DataValues.rescale_epsilon}",
                 "--n_step", f"{DataValues.n_step}",
                 "--num_iterations", f"{DataValues.num_iterations}",
                 "--num_train_steps", f"{DataValues.num_train_steps}",
                 "--num_eval_steps", f"{DataValues.num_eval_steps}",
                 "--target_net_update_interval", f"{DataValues.target_net_update_interval}",
                 "--actor_update_interval", f"{DataValues.actor_update_interval}",
                 "--eval_exploration_epsilon", f"{DataValues.eval_exploration_epsilon}",
                 "--num_actors", f"{DataValues.num_actors}",
                 "--tag", f"{DataValues.tag}",
                 "--environment_height", f"{DataValues.environment_height}",
                 "--environment_width", f"{DataValues.environment_width}",
                 "--environment_frame_skip", f"{DataValues.environment_frame_skip}",
                 "--environment_frame_stack", f"{DataValues.environment_frame_stack}",
                 "--max_episode_steps", f"{DataValues.max_episode_steps}",
                 "--compress_state", f"{DataValues.compress_state}",
                 "--unroll_length", f"{DataValues.unroll_length}",
                 "--use_tensorboard", f"{DataValues.use_tensorboard}",
                 "--debug_screenshots_interval", f"{DataValues.debug_screenshots_interval}",
                 "--results_csv_path", f"{DataValues.results_csv_path}",
                 "--checkpoint_dir", f"{DataValues.checkpoint_dir}",
                 "--burn_in", f"{DataValues.burn_in}",
                 "--actors_on_gpu", f"{DataValues.actors_on_gpu}",
                 ]
    
    # Start experiment_start on a new thread, since it's very light-weight task.
    logging.debug("run_experiment_start_separately")
    thread = threading.Thread(target=run_experiment_start_task, args=(
        DataValues.algorithm, 
        arguments, 
        DataValues.ctx,
        DataValues.data_queue,
        DataValues.shared_params,
        DataValues.start_event,
        DataValues.stop_event,
        DataValues.global_stop_event,
        DataValues.log_queue,
        DataValues.ui_log_queue,
        DataValues.iteration_count,
        DataValues.procs,
        DataValues.experiment_finished_event,
    ), daemon=True)
    thread.start()
    return thread

def run_progress_bar_separately(
    progress_bar_stop_event: multiprocessing.Event,
    iteration_count: multiprocessing.Value
    ):
    print("run_progress_bar_separately")
    # progress_bar_process = DataValues.ctx.Process(
    #     name="progress_bar_job", 
    #     target=run_progress_bar, 
    #     args=(
    #         progress_bar,
    #         stop_event, 
    #         global_stop_event,
    #         iteration_count,
    #         progress_steps
    #     ),
    # )
    # # TODO: Look how to solve this control via multiprocessing context
    # DataValues.procs.append(progress_bar_process)
    # print("run_progress_bar_separately start")
    # progress_bar_process.start()
    # print("run_progress_bar_separately join")
    # progress_bar_process.join()
    # print("run_progress_bar_separately terminate")
    # progress_bar_process.terminate()
    
    thread = threading.Thread(target=run_progress_bar_task, args=(progress_bar_stop_event, iteration_count), daemon=True)
    thread.start()
    return thread
    
def run_statistics_separately(
    statistics_stop_event: multiprocessing.Event,
    ui_log_queue: multiprocessing.Value
):
    # Start statistics on a new thread, since it's very light-weight task.
    logging.debug("run_statistics_separately")
    thread = threading.Thread(target=run_statistics_task, args=(statistics_stop_event, ui_log_queue), daemon=True)
    thread.start()
    return thread

def run_experiment_finished_handler_separately(
    experiment_finished_event: multiprocessing.Event
):
    # Start statistics on a new thread, since it's very light-weight task.
    logging.debug("run_experiment_finished_handler_separately")
    thread = threading.Thread(target=run_experiment_finished_task, args=(experiment_finished_event,), daemon=True)
    thread.start()
    return thread

def run_tensorboard_separately(
    tensorboard_url: multiprocessing.Value,
    tensorboard_thread_running: multiprocessing.Value
):
    if tensorboard_thread_running:
        logging.debug("Tensorboard still running")
        return tensorboard_thread.value
    else:
        # Start statistics on a new thread, since it's very light-weight task.
        logging.debug("run_tensorboard_separately")
        thread = threading.Thread(target=run_tensorboard_task, args=(tensorboard_url, tensorboard_thread_running), daemon=True)
        thread.start()
        tensorboard_thread_running.set()
        return thread

def started_state():
    dpg.disable_item("start_button")
    dpg.hide_item("start_button")
    dpg.enable_item("stop_button")
    dpg.enable_item("tensorboard_button_v2")
    dpg.show_item("stop_button")
    dpg.show_item("tensorboard_button_v2")
    dpg.show_item("progress_bar")
    dpg.show_item("current_statistics")
    dpg.show_item("max_statistics")
    dpg.show_item("min_statistics")
    
def start_experiment(sender, value, app_data, user_data):
    DataValues.experiment_finished_event.clear()
    
    started_state()
    
    print(f"sender is: {sender}")
    print(f"value is: {value}")
    print(f"app_data is: {app_data}")
    print(f"user_data is: {user_data}")
    
    DataValues.update_tag()
    
    threads = []
    
    experiment_start_thread = run_experiment_start_separately()
    threads.append(experiment_start_thread)
    
    progress_bar_thread = run_progress_bar_separately(
        DataValues.progress_bar_stop_event,
        DataValues.iteration_count
    )
    threads.append(progress_bar_thread)
    
    statistics_thread = run_statistics_separately(
        DataValues.statistics_stop_event,
        DataValues.ui_log_queue
    )
    threads.append(statistics_thread)
    
    experiment_finished_handler_thread = run_experiment_finished_handler_separately(
        DataValues.experiment_finished_event
    )
    threads.append(experiment_finished_handler_thread)

def handle_experiment_finished():
    logging.debug("Main: before progress_bar_stop_event.set()")
    DataValues.progress_bar_stop_event.set()
    DataValues.statistics_stop_event.set()

    logging.debug("Main: before stopped_state()")
    stopped_state()
    
def setup_logging():
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel('DEBUG')
    
def main():
    setup_logging()
    data = DataValues()
    
    # multiprocessing.set_start_method('spawn')
    DataValues.ctx = multiprocessing.get_context("spawn")
    (DataValues.child, pipe) = DataValues.ctx.Pipe(duplex=True)
    # print(multiprocessing.get_start_method())
    
    logging.info('multiprocessing.Manager()')
    # Create shared objects so all actor processes can access them
    DataValues.manager = multiprocessing.Manager()
    
    logging.info(f'FLAGS.num_actors: {DataValues.num_actors}')
    # Create queue to shared transitions between actors and learner
    DataValues.data_queue = DataValues.manager.Queue(maxsize=DataValues.num_actors * 2)
    
    # logging.info(f'FLAGS.num_actors: {DataValues.num_actors}')
    # # Create queue to shared transitions between actors and learner
    # DataValues.data_queue = multiprocessing.Queue(maxsize=DataValues.num_actors * 2)
    
    logging.info('Before manager.dict()')
    # Store copy of latest parameters of the neural network in a shared dictionary, so actors can later access it
    DataValues.shared_params = DataValues.manager.dict({'network': None})
    
    DataValues.start_event = DataValues.manager.Event()
    DataValues.stop_event = DataValues.manager.Event()
    DataValues.global_stop_event = DataValues.manager.Event()
    DataValues.progress_bar_stop_event = DataValues.manager.Event()
    DataValues.statistics_stop_event = DataValues.manager.Event()
    DataValues.experiment_finished_event = DataValues.manager.Event()
    DataValues.log_queue = DataValues.manager.Queue() # May be SimpleQueue
    DataValues.ui_log_queue = DataValues.manager.Queue()
    DataValues.iteration_count = DataValues.manager.Value('i', 0) # May be DataValues.manager.Value(c_int, 0)
    DataValues.tensorboard_url = DataValues.manager.Value(c_char_p, "")
    DataValues.tensorboard_thread_running = DataValues.manager.Event()

    dpg.create_context()
    dpg.create_viewport(title='Reinforcement learning on games', width=860, height=1000)
    
    window = dpg.window(label="Learn start window", width=850, height=990)
    print(f"window: {window}")

    with window:
        with dpg.tab_bar(tag="main_tab_bar") as tb:
            with dpg.tab(label="learning", tag="learning_tab"):
                dpg.add_combo(label="Select algorithm", callback=update_algorithm, items=["r2d2", "PPO", "A2C", "DQN", "atari57"], default_value="r2d2")
                dpg.add_text("Set environment:")
                dpg.add_combo(label="Select type", items=["Atari", "Classic"], default_value="Atari")
                dpg.add_combo(label="Select image source", callback=update_image_source, items=["ALE", "Shimmy"], default_value="Shimmy")
                
                dpg.add_combo(label="Select environment name", callback=update_environment_name, items=DataValues.environment_names, default_value="Pong", tag="EnvNameSelector")
                
                dpg.add_input_int(label="Random seed", callback=update_seed, default_value=DataValues.seed)
                dpg.add_input_int(label="Frame skip", callback=update_environment_frame_skip, default_value=DataValues.environment_frame_skip)
                dpg.add_input_int(label="Frame stack", callback=update_environment_frame_stack, default_value=DataValues.environment_frame_stack)
                
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Compress state", callback=update_compress_state, default_value=DataValues.compress_state)
                    dpg.add_checkbox(label="Clip gradients", callback=update_clip_grad, default_value=DataValues.clip_grad)
                    
                dpg.add_input_int(label="Number of actors", callback=update_num_actors, default_value=DataValues.num_actors)
                dpg.add_input_int(label="Replay buffer capacity", callback=update_replay_capacity, default_value=DataValues.replay_capacity)
                dpg.add_input_int(label="Minimal replay buffer size", callback=update_min_replay_size, default_value=DataValues.min_replay_size)
                dpg.add_input_int(label="Max gradients norm", callback=update_max_grad_norm, default_value=DataValues.max_grad_norm)
                
                dpg.add_text("Set hyperparameters (ctrl+click for manual digits input):")
                
                dpg.add_slider_float(label="Learning rate", callback=update_learning_rate, default_value=DataValues.learning_rate, format=f"%.4f", min_value=0.00001, max_value=1)
                dpg.add_slider_float(label="Adam epsilon", callback=update_adam_eps, default_value=DataValues.adam_eps, format=f"%.4f", min_value=0.00001, max_value=1)
                dpg.add_slider_float(label="Discount", callback=update_discount, default_value=DataValues.discount, format=f"%.4f", max_value=1)
                
                dpg.add_input_int(label="Unroll length", callback=update_unroll_length, default_value=DataValues.unroll_length)
                dpg.add_input_int(label="Burn in", callback=update_burn_in, default_value=DataValues.burn_in)
                
                batch_size_items_int = 2 ** np.arange(14)
                batch_size_items = batch_size_items_int.astype(str).tolist()
                # batch_size_items = np.char.mod('%d', batch_size_items_int).tolist()
                dpg.add_combo(label="Batch size", callback=update_batch_size, items=batch_size_items, default_value=DataValues.batch_size, tag='batch_size')
                dpg.add_slider_float(label="Priority exponent", callback=update_priority_exponent, default_value=DataValues.priority_exponent, format=f"%.4f", min_value=0.00001, max_value=1)
                dpg.add_slider_float(label="Importance sampling exponent", callback=update_importance_sampling_exponent, default_value=DataValues.importance_sampling_exponent, format=f"%.4f", min_value=0.00001, max_value=1)
                
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Normalize weights", callback=update_normalize_weights, default_value=DataValues.normalize_weights)
                    dpg.add_checkbox(label="GPU actors", callback=update_gpu_actors, default_value=DataValues.actors_on_gpu)
                    dpg.add_checkbox(label="Use tensorboard", callback=update_use_tensorboard, default_value=DataValues.use_tensorboard)
                
                dpg.add_slider_float(label="Priority eta", callback=update_priority_eta, default_value=DataValues.priority_eta, format=f"%.4f", min_value=0.00001, max_value=1)
                dpg.add_slider_float(label="Rescale epsilon", callback=update_rescale_epsilon, default_value=DataValues.rescale_epsilon, format=f"%.4f", min_value=0.00001, max_value=1)
                dpg.add_input_int(label="TD n-step bootstrap", callback=update_n_step, default_value=DataValues.n_step)
                dpg.add_input_int(label="Number of iterations to run", callback=update_num_iterations, default_value=DataValues.num_iterations)
                dpg.add_input_int(label="Number of training steps", callback=update_num_train_steps, default_value=DataValues.num_train_steps)
                
                dpg.add_input_int(label="Number of evaluation steps", callback=update_num_eval_steps, default_value=DataValues.num_eval_steps)
                dpg.add_input_int(label="Maximum steps (before frame skip) per episode", callback=update_max_episode_steps, default_value=DataValues.max_episode_steps)
                dpg.add_input_int(label="Target Q networks update interval", callback=update_target_net_update_interval, default_value=DataValues.target_net_update_interval)
                dpg.add_input_int(label="Local Q network update interval", callback=update_actor_update_interval, default_value=DataValues.actor_update_interval)
                dpg.add_slider_float(label="Fixed exploration rate in e-greedy policy for evaluation", callback=update_eval_exploration_epsilon, default_value=DataValues.eval_exploration_epsilon, format=f"%.4f", min_value=0.00001, max_value=1)
                
                dpg.add_input_text(label="Experiment tag", default_value=DataValues.tag, tag='tag')
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Start", callback=start_experiment, tag="start_button")
                    dpg.add_button(label="Stop", callback=stop_experiment, enabled=False, tag="stop_button")
                    dpg.hide_item("stop_button")
                    dpg.add_button(
                        label="Go to Tensorboard", 
                        callback=open_tensorboard_with_filter_v2, 
                        enabled=False, 
                        tag="tensorboard_button_v2", 
                        user_data=DataValues.timestr
                    )
                    dpg.hide_item("tensorboard_button_v2")
                dpg.add_progress_bar(label="progress", default_value=0.0, tag="progress_bar")
                dpg.hide_item("progress_bar")
                
                with dpg.group(horizontal=True, tag="current_statistics"):
                    dpg.add_text("Current: iteration ") 
                    dpg.add_text("", tag="current_iteration")
                    dpg.add_text(" step ") 
                    dpg.add_text("", tag="current_step")
                dpg.hide_item("current_statistics")
                    
                with dpg.group(horizontal=True, tag="max_statistics"):
                    # iteration:  13, role: R2D2-actor5, step: 6500000, episode_return:  5418.03, num_episodes: 646, step_rate:  143, duration: 3502.60
                    dpg.add_text("Max: episode_return ")
                    dpg.add_text("", tag="max_episode_return")
                    dpg.add_text(" step_rate ")
                    dpg.add_text("", tag="max_step_rate")
                    dpg.add_text(" duration ")
                    dpg.add_text("", tag="max_duration")
                dpg.hide_item("max_statistics")
                
                with dpg.group(horizontal=True, tag="min_statistics"):
                    # iteration:  13, role: R2D2-actor5, step: 6500000, episode_return:  5418.03, num_episodes: 646, step_rate:  143, duration: 3502.60
                    dpg.add_text("Min: episode_return ")
                    dpg.add_text("", tag="min_episode_return")
                    dpg.add_text(" step_rate ")
                    dpg.add_text("", tag="min_step_rate")
                    dpg.add_text(" duration ")
                    dpg.add_text("", tag="min_duration")
                dpg.hide_item("min_statistics")
                
                # dpg.add_button(label="Go to experiments history", callback=change_tab, tag=100)
            with dpg.tab(label="experiments history", tag="experiments_history"):
                if (exists("runs")):
                    dpg.add_button(label="Go to tensorboard!", callback=open_tensorboard, tag="tensorboard_button")
                
                history_table()
                                
                #creating a button that executes the callback change_tab with the tag 200
                dpg.add_button(label="Go to learning process", tag=200, callback=change_tab)
    
    
    
# def open_tensorboardA(sender, value, app_data, user_data):
#     print(f"sender is: {sender}")
#     print(f"value is: {value}")
#     print(f"app_data is: {app_data}")
#     print(f"user_data is: {user_data}")
#     import subprocess

#     some = subprocess.run(["tensorboard", "--logdir", "runs"]) 
#     print(some)
    
#     import webbrowser

#     url = "http://localhost:6006/"

#     webbrowser.open(url, new=0, autoraise=True)

def open_tensorboard(sender, value, app_data, user_data):
    tracking_address = "runs" # the path of your log file.

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    
    import webbrowser
    webbrowser.open(url, new=0, autoraise=True)
    
def open_tensorboard_with_filter_v2(sender, value, app_data, user_data):
    print(f"sender is: {sender}")
    print(f"value is: {value}")
    print(f"app_data is: {app_data}")
    print(f"user_data is: {user_data}")
    
    tracking_address = "runs" # the path of your log file.

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    url += f"?runFilter={user_data}"
    print(f"Tensorflow listening on {url}")
    
    import webbrowser
    webbrowser.open(url, new=0, autoraise=True)
    
def open_tensorboard_with_filter(sender, value, app_data, user_data):
    run_tensorboard_separately(DataValues.tensorboard_url, DataValues.tensorboard_thread)
    DataValues.tensorboard_url += "?runFilter=163922"
    print(f"Tensorflow listening on {DataValues.tensorboard_url}")
    
    import webbrowser
    webbrowser.open(DataValues.tensorboard_url, new=0, autoraise=True)

def change_tab(sender):
    #if the sender is the tag 100 ( Button 1 ) we set the value of the main_tab_bar to the experiments_history
    #if the sender isn't the tag 100 ( Button 1) we set the value of the main_tab_bar to the learning_tab
    if sender == 100:
        dpg.set_value("main_tab_bar", "experiments_history")
    else:
        dpg.set_value("main_tab_bar", "learning_tab")
        
def get_experiments():
    import os
    experiments = os.listdir("runs")
    experiment_list = run_folders_to_list(experiments)
    from typing import Dict, List
    experiments_by_date_time: Dict[str, list] = dict()
    
    # from dataclasses import dataclass

    # @dataclass
    # class ExperimentActor:
    #     env_name: str
    #     env_version: str
    #     algo: str
    #     actor_name: str
    #     hyperparameters: str
    
    for actor in experiment_list:
        env_name = actor[0]
        env_property = ''
        no_frameskip = "NoFrameskip"
        if no_frameskip in env_name:
            env_property = no_frameskip
            env_name = env_name.replace(no_frameskip, '')
        date = actor[4]
        time = actor[5]
        experiment_actor = {
            "env_name": env_name,
            "env_property": env_property,
            "env_version": actor[1],
            "algo": actor[2],
            "actor_name": actor[3],
            "date": date,
            "time": time,
            "hyperparameters": actor[6:],
        }
        # experiment_actor2 = ExperimentActor(
        #     env_name = actor[0],
        #     env_version = actor[1],
        #     algo = actor[2],
        #     actor_name = actor[3],
        #     hyperparameters = actor[6:]
        # )
        date_time = f"{date}-{time}"
        if date_time not in experiments_by_date_time:
            singleton: List[Dict] = list()
            singleton.append(experiment_actor)
            experiments_by_date_time[date_time] = singleton
        else:
            curr_list: List[Dict] = experiments_by_date_time[date_time]
            curr_list.append(experiment_actor)
    return experiments_by_date_time

def run_folders_to_list(experiments):
    experiment_list = list()
    import re
    for name in experiments:
        left = 0
        experiment_details = list()
        for match in re.finditer('-', name):
            start, right = match.regs[0]
            element = name[left:right-1]
            experiment_details.append(element)
            left = right
        experiment_list.append(experiment_details)
    return experiment_list
        
def experiments_to_table(experiments):
    table = []
    for key, value in experiments.items():
        first_element = value[0]
        env_name = first_element["env_name"]
        tag = first_element["hyperparameters"]
        date_time = f'{first_element["date"]}-{first_element["time"]}'
        row = [env_name, date_time, tag]
        table.append(row)
     
    return table
    
def history_table():
    experiments_by_date_time = get_experiments()
    matrix = experiments_to_table(experiments_by_date_time)
    with dpg.table(header_row=True, policy=dpg.mvTable_SizingStretchProp, reorderable=True, resizable=True, row_background=True,
                   borders_innerH=True, borders_outerH=True, borders_innerV=True,
                   borders_outerV=True, no_host_extendX=True):
                # use add_table_column to add columns to the table,
                # table columns use child slot 0
        dpg.add_table_column(label="Environment", width_stretch=True, init_width_or_weight=0.0)
        dpg.add_table_column(label="Date-Time", width_stretch=True, init_width_or_weight=0.0)
        dpg.add_table_column(label="Tag", width_stretch=True, init_width_or_weight=0.0)
        # dpg.add_table_column(label="Max Avg Reward")

                # add_table_next_column will jump to the next row
                # once it reaches the end of the columns
                # table next column use slot 1
        rows = matrix
        for row in rows:
            with dpg.table_row():
                for column in row:
                    dpg.add_button(label=column, callback=open_tensorboard_with_filter, user_data=column)


            
# demo.show_demo()

if __name__ == "__main__":
    main()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    
    # multiprocessing.set_start_method('spawn')
    # sm = multiprocessing.get_start_method()
    # if (sm == 'fork'):
    #     multiprocessing.set_start_method('spawn')
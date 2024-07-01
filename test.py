import dearpygui.dearpygui as dpg
# from dearpygui import core as dpgc

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=600, height=300)

def callback():
    count: float = dpg.get_value("progress")
    count = count + 0.01
    if count >= 1.0:
        count = 0.0
    dpg.set_value("progress", count)


with dpg.window(label="Example Window"):
    dpg.add_progress_bar(tag="progress", overlay="status")

    # dpgc.set_render_callback("callback")
    # dpg.add_text("Hello, world")
    # dpg.add_button(label="Save")
    # dpg.add_input_text(label="string", default_value="Quick brown fox")
    # dpg.add_slider_float(label="float", default_value=0.273, max_value=1)


        
dpg.setup_dearpygui()
dpg.show_viewport()
# dpg.start_dearpygui()

while dpg.is_dearpygui_running():
    # insert here any code you would like to run in the render loop
    # you can manually stop by using stop_dearpygui()
    # print("this will run every frame")
    callback()
    dpg.render_dearpygui_frame()

dpg.destroy_context()
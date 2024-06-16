import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
from modules.rembg import rembg_run
import args_manager
import copy
import launch

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

def get_task(*args):
    args = list(args)
    args.pop(0)

    return worker.AsyncTask(args=args)

def generate_clicked(task):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    # outputs=[progress_html, progress_window, progress_gallery, gallery]
    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False)

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False)
            if flag == 'finish':
                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True

    execution_time = time.perf_counter() - execution_start_time
    global time_taken 
    time_taken = f"Total time: {execution_time:.2f} seconds"
    print(time_taken)
    return

reload_javascript()

title = f'fashionCORE {fooocus_version.version}'

shared.gradio_root = gr.Blocks(
    title=title,
    css=modules.html.css,
    theme="ehristoforu/Indigo_Theme").queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("Generation"):
                with gr.Row():
                    progress_window = grh.Image(label='Preview', show_label=True, visible=False, height=768,
                                                elem_classes=['main_view'])
                    progress_gallery = gr.Gallery(label='Finished Images', show_label=True, object_fit='contain',
                                                  height=768, visible=False, elem_classes=['main_view', 'image_gallery'])
                progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                        elem_id='progress-bar', elem_classes='progress-bar')
                gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', visible=True, height=768,
                                     elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                     elem_id='final_gallery')

            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here or paste parameters.", elem_id='positive_prompt',
                                        container=False, autofocus=True, elem_classes='type_row', lines=1024)
                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row', elem_id='generate_button', visible=True)
                    load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', visible=False)
                    stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half', elem_id='stop_button', visible=False)

            with gr.Row(visible=True) as image_input_panel:
                with gr.Tabs():
                    with gr.TabItem(label='Clothes/Accessories') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for _ in range(flags.controlnet_image_count):
                                with gr.Column():
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False, height=300)
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=True) as ad_col:
                                        with gr.Row():
                                            default_end, default_weight = flags.default_parameters[flags.default_ip]

                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0, step=0.001, value=0.85)
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0, step=0.001, value=0.975)
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=flags.default_ip, container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                            
                    with gr.TabItem(label='Person Image') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = grh.Image(label='Drag person image here', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas')
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.flags.inpaint_option_default, label='Method', visible=False)

        with gr.Column(scale=1, visible=True):
            with gr.Tab(label='Settings'):
                performance_selection = gr.Radio(label='Performance',
                                                 choices=modules.flags.performance_selections,
                                                 value=modules.config.default_performance,
                                                 elem_classes='performance_selections')
                preset_selection = gr.Radio(label='Preset',
                                            choices=['realistic'],
                                            value='realistic',
                                            interactive=True)
                
                aspect_ratios_selection = gr.Radio(label='Aspect Ratios', choices=['1024×1024', '1152×896', '1344×768', '1408×704'],
                                                   value='1152×896', info='width × height',
                                                   elem_classes='aspect_ratios')

                image_number = gr.Slider(label='Image Number', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)

                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.",
                                             info='Describing what you do not want to see.', lines=2,
                                             elem_id='negative_prompt',
                                             value=modules.config.default_prompt_negative)

                overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                           minimum=-1, maximum=200, step=1,
                                           value=modules.config.default_overwrite_step,
                                           info='Set as -1 to disable. For developer debugging.')

            with gr.Tab(label='Styles'):
                style_sorter.try_load_sorted_styles(
                    style_names=legal_style_names,
                    default_selected=[
                        "Fooocus V2",
                        "Fooocus Photograph",
                        "Fooocus Negative",
                        "Fooocus Masterpiece",
                        "Fooocus Enhance",
                        "Fooocus Sharp",
                        "Fooocus Cinematic",
                        "SAI Analog Film"
                    ])

                style_search_bar = gr.Textbox(show_label=False, container=False,
                                              placeholder="\U0001F50E Type here to search styles ...",
                                              value="",
                                              label='Search Styles')
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=copy.deepcopy(style_sorter.all_styles),
                                                    value=copy.deepcopy(style_sorter.all_styles[:8]),
                                                    label='Selected Styles',
                                                    elem_classes=['style_selections'])
                gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)

                shared.gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                        outputs=style_selections)

                style_search_bar.change(style_sorter.search_styles,
                                        inputs=[style_selections, style_search_bar],
                                        outputs=style_selections,
                                        queue=False,
                                        show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

                gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                       inputs=style_selections,
                                                       outputs=style_selections,
                                                       queue=False,
                                                       show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

        state_is_generating = gr.State(False)

        load_data_outputs = [image_number, prompt, negative_prompt, style_selections,
                             performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, base_model,
                             refiner_model, refiner_switch, sampler_name, scheduler_name, seed_random, image_seed,
                             generate_button, load_parameter_button] + freeu_ctrls + lora_ctrls

        preset_selection.change(lambda preset: modules.meta_parser.load_parameter_button_click(preset, state_is_generating),
                                inputs=preset_selection, outputs=load_data_outputs, queue=False, show_progress=False)

        performance_selection.change(lambda x: [gr.update(interactive=x != 'Extreme Speed')] * 11 +
                                               [gr.update(visible=x != 'Extreme Speed')] * 1 +
                                               [gr.update(interactive=x != 'Extreme Speed', value=x == 'Extreme Speed', )] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                     ], queue=False, show_progress=False)
        
        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, translate_prompts, style_selections,
            performance_selection, aspect_ratios_selection, image_number, output_format, image_seed, sharpness, guidance_scale
        ]
        ctrls += [inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += [disable_preview, disable_intermediate_results, black_out_nsfw]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg]
        ctrls += [sampler_name, scheduler_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += freeu_ctrls
        ctrls += ip_ctrls

        generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True),
                              outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery]) \
            .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
                  outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
            .then(fn=update_history_link, outputs=history_link)

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    favicon_path="assets/favicon.png",
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    blocked_paths=[constants.AUTH_FILENAME]
    )
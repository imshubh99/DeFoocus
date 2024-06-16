import gradio as gr
import random
import os
import json
import time
import shared
import modules.config as config
import fooocus_version
import modules.html as html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser as meta_parser
from modules.rembg import rembg_run
import args_manager
import copy
import launch
from SegBody import segment_body

def get_task(*args):
    args = list(args)
    args.pop(0)
    return worker.AsyncTask(args=args)

def generate_clicked(clothing_image, person_image, performance_selection, preset_selection, style_selections, segmentation_mask):
    pass

def generate_segmentation_mask(image):
    seg_image, mask_image = segment_body(image, face=False)
    mask_image = mask_image.resize((512, 512))
    return mask_image

title = "FashionCore"
description = "Try on different clothes and accessories on your images."

with gr.Blocks(title=title, css=config.css) as demo:
    gr.Markdown(description)
    
    with gr.Tab("Free Version"):
        with gr.Row():
            with gr.Column():
                clothing_image = gr.Image(label="Clothing Image", source="upload", type="numpy")
                person_image = gr.Image(label="Person Image", source="upload", type="numpy", tool="sketch", elem_id="person_image")
                segmentation_mask = gr.Image(label="Segmentation Mask", visible=False)
                with gr.Accordion("Settings", open=False):
                    performance_selection = gr.Radio(label="Performance", choices=flags.performance_selections, value="Quality")
                    preset_selection = gr.Radio(label="Preset", choices=["Realistic"], value="Realistic")
                    style_selections = gr.CheckboxGroup(label="Styles", choices=[
                        "Fooocus V2",
                        "Fooocus Photograph",
                        "Fooocus Negative",
                        "Fooocus Masterpiece",
                        "Fooocus Enhance",
                        "Fooocus Sharp",
                        "Fooocus Cinematic",
                        "SAI Analog Film"
                    ], value=[
                        "Fooocus V2",
                        "Fooocus Photograph",
                        "Fooocus Negative",
                        "Fooocus Masterpiece", 
                        "Fooocus Enhance",
                        "Fooocus Sharp",
                        "Fooocus Cinematic",
                        "SAI Analog Film"
                    ])
                with gr.Accordion("Advanced Settings", open=False):
                    auto_mask_checkbox = gr.Checkbox(label="Use Auto Mask", value=True)
                    debug_mode_checkbox = gr.Checkbox(label="Debug Mode", value=False)
            
            with gr.Column():
                generated_image = gr.Image(label="Generated Image", visible=False)
                generate_button = gr.Button("Generate")
        
        def generate_mask_clicked(person_image):
            if auto_mask_checkbox.value and debug_mode_checkbox.value:
                return generate_segmentation_mask(person_image)
            return None
        
        person_image.change(fn=generate_mask_clicked, inputs=person_image, outputs=segmentation_mask)
        generate_button.click(fn=generate_clicked, inputs=[clothing_image, person_image, performance_selection, preset_selection, style_selections, segmentation_mask], outputs=generated_image)
    
    with gr.Tab("Paid Version"):
        gr.Markdown("Upgrade to the paid version to access advanced features and models.")
        
        with gr.Accordion("Advanced Features", open=False):
            gr.Markdown("- Photopea integration for advanced editing")
            gr.Markdown("- Additional advanced models and presets")
            gr.Markdown("- Enhanced user interface and experience")
    
    with gr.Tab("Remove Background"):
        with gr.Row():
            rembg_input = grh.Image(label="Input Image", source="upload", type="filepath")
            rembg_button = gr.Button("Remove Background")
        with gr.Row():
            rembg_output = grh.Image(label="Output Image")
        rembg_button.click(rembg_run, inputs=rembg_input, outputs=rembg_output)
    
    with gr.Tab("Segmentation Mask"):
        with gr.Row():
            segmentation_input = gr.Image(label="Input Image", source="upload", type="pil")
            segmentation_button = gr.Button("Generate Mask")
        with gr.Row():
            segmentation_output = gr.Image(label="Segmentation Mask")
        segmentation_button.click(generate_segmentation_mask, inputs=segmentation_input, outputs=segmentation_output)

demo.launch(inbrowser=args_manager.args.in_browser,
            server_name=args_manager.args.listen,
            server_port=args_manager.args.port,
            share=args_manager.args.share,
            favicon_path="assets/favicon.png",
            auth=None,
            blocked_paths=[constants.AUTH_FILENAME])
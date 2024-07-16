import os
import numpy as np
import gradio as gr
from einops import rearrange

from glsim.other_utils.build_args import parse_inference_args

from inference import prepare_img, prepare_inference, inference_single
from glsim.train_utils.misc_utils import inverse_normalize


def demo(file_path, dynamic_top=8, dfsm='glsim_norm', dataset='dafb'):
    args = parse_inference_args()

    args.vis_mask = dfsm
    if args.vis_mask == 'glsim_norm':
        args.vis_mask_sq = True

    if dataset == 'dafb':
        norm_custom = True
        args.dynamic_top = dynamic_top
        args.ckpt_path = 'ckpts/dafb_glsim.pth'
        args.dataset_root_path = os.path.join('data', 'daf')
    elif dataset == 'inat17':
        norm_custom = False
        args.dynamic_top = dynamic_top
        args.ckpt_path = 'ckpts/inat_glsim.pth'
        args.dataset_root_path = os.path.join('data', 'inat17')
    elif dataset == 'nabirds':
        norm_custom = False
        args.dynamic_top = dynamic_top * 2
        args.ckpt_path = 'ckpts/nabirds_glsim.pth'
        args.dataset_root_path = os.path.join('data', 'nabirds')

    model, transform, dic_classid_classname = prepare_inference(args)

    img = prepare_img(file_path, args, transform)

    top1_text, images_crops, masked_image = inference_single(args, model, img, dic_classid_classname)

    images_crops = inverse_normalize(images_crops.data, norm_custom)
    images_crops = rearrange(images_crops, '1 c h w -> h w c')
    images_crops = np.uint8(np.clip(images_crops.to('cpu').numpy(), 0, 1) * 255)

    masked_image = np.array(masked_image)

    return top1_text, images_crops, masked_image


title = 'GLSim'
description = 'Global-Local Similarity for Efficient Fine-Grained Image Recognition with Vision Transformers'
article = '''<p style='text-align: center'>
    <a href='https://github.com/arkel23/GLSim/'>
    Global-Local Similarity for Efficient Fine-Grained Image Recognition with Vision Transformers</a> | 
    <a href='https://github.com/arkel23/GLSim/'>GitHub Repo</a></p>'''

inputs = [
    gr.components.Image(type='filepath', label='Input image'),
    gr.components.Radio(value=8, choices=[1, 2, 4, 8, 16, 32, 64],
                        label='Top-O tokens for similarity selection (def: 8)'),
    gr.components.Radio(value='glsim_norm', choices=['glsim_norm', 'rollout'],
                        label='For visualizing DFSM criteria'),
    gr.components.Radio(value='dafb', choices=['dafb', 'inat17', 'nabirds'],
                        label='Dataset (def: dafb)'),
]

outputs = [
    gr.components.Textbox(label='Predicted class and tags'),
    gr.components.Image(label='Crop'),
    gr.components.Image(label='DFSM Criteria')
]

examples = [
    ['samples/others/dafb_rena_170785.jpg'],
    ['samples/inat/notarctia_proxima_add650.jpg'],
    ['samples/others/cub_common_yellowthroat.jpg'], 
    ]

gr.Interface(
    demo, inputs, outputs, title=title, description=description,
    article=article, examples=examples).launch(debug=True, share=True)

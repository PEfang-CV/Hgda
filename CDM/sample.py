"""
Function: Generate defect image by CDM 
Author: TyFang
Date: 2023/12/29
"""
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import cv2
import conf_mgt
from utils import yamlread
from utils import imwrite
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

def toU8(sample):
    if sample is None:
        return sample
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def show_images(imgs, img_names, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for image_name, image in zip(img_names, imgs):
        out_path = os.path.join(dir_path, image_name)
        image,g,r=cv2.split(image)
        cv2.imshow("Results",image)
        # imwrite(img=image, path=out_path)

def main(conf: conf_mgt.Default_Conf):
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    pretrain_model=dist_util.load_state_dict(args.checkmodel_path, map_location="cpu")
    model.load_state_dict(pretrain_model,strict=False)


    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    cond_fn = None
    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y)

    logger.log("sampling...")
    all_images = []
    dset = 'eval'
    eval_name = conf.get_default_eval_name()
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(dist_util.dev())

        cond_kwargs = {}
        cond_kwargs["gt"] = batch['GT']
        gt_keep_mask = batch.get('gt_keep_mask')
        cond_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = cond_kwargs["gt"].shape[0]
        classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev())


        model_kwargs= {}
        model_kwargs["y"] = batch.get('gt_keep_mask')

        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)

        result = sample_fn(
            model_fn,
            (batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            #cond_kwargs=cond_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev()
        )
        result=toU8(result)

        show_images(result,batch['GT_name'], args.out_file)
    print("sampling complete")



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        use_ddim=True,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        timestep_respacing="ddim50",
        checkmodel_path="",
        out_file=""
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread('./confs/NEU-Seg.yml'))
    main(conf_arg)

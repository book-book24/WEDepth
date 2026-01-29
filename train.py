import os
import json
import argparse
import warnings
import logging
import pprint
import random
import time


import dataset as DS
import numpy as np
from util.utils import init_log
from util.dist_helper import setup_distributed
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
# from torch.optim.lr_scheduler import LinearLR
from util.lr_scheduler import WarmupCosineSchedule
from module.Model import WEDepth
from util.loss import SiLogLoss
from util.metric import eval_depth


def main(config, args):


    # warnings.simplefilter('ignore', np.RankWarning) # 用于过滤警告
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))


    cudnn.enabled = True
    cudnn.benchmark = True

    trainset = getattr(DS, config["data"]["train_dataset"]["name"])(test_mode=False, base_path=config["data"]["train_dataset"]["root"], crop="garg", augmentations_db=config["data"]["augmentations"])
    trainsampler = DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=config["training"]["batch_size"], pin_memory=True, num_workers=4, drop_last=True,
                             sampler=trainsampler)

    val_loaders = []
    for _val_dic in config["data"]["val_dataset"]:

        valset = getattr(DS, _val_dic["name"])(test_mode=True, base_path=_val_dic["root"])
        valsampler = DistributedSampler(valset)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
        val_loaders.append(valloader)

    local_rank = int(os.environ["LOCAL_RANK"])

    # 调用模型
    model = WEDepth(config)


    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    if rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
            model_total_params, model_grad_params))
        logger.info("tuned percent:%.3f" % (model_grad_params / model_total_params * 100))


    criterion1 = SiLogLoss().cuda(local_rank)
    optimizer = AdamW(model.parameters(), lr=config["training"]["lr"], betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = WarmupCosineSchedule(optimizer,warmup_steps=config["training"]["warm_up"]*len(trainloader), t_total=config["training"]["epochs"]*len(trainloader))

    scaler = torch.cuda.amp.GradScaler()
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config["training"]["f16"])

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}

    for epoch in range(config["training"]["epochs"]):

        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, config["training"]["epochs"], previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, config["training"]["epochs"], previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'],
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))

        trainloader.sampler.set_epoch(epoch + 1)

        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            with context:
                img, depth, valid_mask = sample['image'].cuda(), sample['gt'].cuda(), sample['mask'].cuda()
                pred = model(img, valid_mask, config["data"]["train_dataset"]["name"])
                loss = criterion1(pred, depth, valid_mask.bool())

            if config["training"]["f16"]:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            total_loss += loss.item()


            if rank == 0 and i % 100 == 0:
                logger.info(
                    'Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'],
                                                                   loss.item()))


        model.eval()

        for idx, valloader in enumerate(val_loaders):
            results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(),
                       'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(),
                       'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
            nsamples = torch.tensor([0.0]).cuda()
            dataset_name = valloader.dataset.name
            dep_min = valloader.dataset.min_depth
            dep_max = valloader.dataset.max_depth

            for i, sample in enumerate(valloader):
                img, depth, valid_mask = sample['image'].cuda(), sample['gt'].cuda(), sample['mask'].cuda()

                with torch.no_grad():
                    pred = model(img,valid_mask,config["data"]["val_dataset"][idx]["name"])
                    pred = torch.clamp(pred, dep_min, dep_max)


                valid_mask = valid_mask.bool()
                if valid_mask.sum() < 10:
                    continue

                cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

                for k in results.keys():
                    results[k] += cur_results[k]
                nsamples += 1

            torch.distributed.barrier()

            for k in results.keys():
                dist.reduce(results[k], dst=0)

            dist.reduce(nsamples, dst=0)

            if rank == 0:
                logger.info('val_set: {:>8}'.format(dataset_name))
                logger.info('==========================================================================================')
                logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
                logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
                logger.info('==========================================================================================')
                print()


            if rank == 0 and (results['d1'] / nsamples).item() >= previous_best['d1']:
                torch.save(model.state_dict(), "best_d1.pth")


            for k in results.keys():
                if k in ['d1', 'd2', 'd3']:
                    previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
                else:
                    previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training script", conflict_handler="resolve"
    )

    parser.add_argument("--config-file", type=str, default="./configs/nyu1.json")
    parser.add_argument("--base-path", default=os.environ.get("TMPDIR", ""))
    parser.add_argument('--port', default=None, type=int)
    parser.add_argument('--local-rank', default=0, type=int)

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    seed = config["generic"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.enabled = True

    main(config, args)


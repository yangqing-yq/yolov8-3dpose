import os
import yaml
import argparse
# from config import cfg
import torch
from base import Trainer, Tester
import torch.backends.cudnn as cudnn
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--parts', type=str, dest='parts')
    parser.add_argument('--continue_train', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"
 
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.parts, 'Please enter human parts among [body, hand, face]'
    return args

def main():
    # argument parse and create log
    args = parse_args()
    cudnn.benchmark = True
    if args.parts == 'body':
        pretrained_weight = 'weights/body_pose_resnet_50_256x192.pth.tar'
        yaml_file = 'configs/body.yaml'
    elif args.parts == 'hand':
        pretrained_weight = 'weights/resnet50.pth'
        yaml_file = 'configs/hand_test.yaml'

    with open(yaml_file,encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        
    # make folder
    os.makedirs(os.path.join(configs['output_dir'], configs['save_name']), exist_ok=True)
    os.makedirs(os.path.join(configs['output_dir'], configs['save_name'], configs['model_dir']), exist_ok=True)
    os.makedirs(os.path.join(configs['output_dir'], configs['save_name'], configs['log_dir']), exist_ok=True)
    os.makedirs(os.path.join(configs['output_dir'], configs['save_name'], configs['result_dir']), exist_ok=True) 
    
    trainer = Trainer(configs)
    trainer._make_batch_generator()
    # trainer._make_model(pretrained_weight, args.parts)
    trainer._make_model_new(pretrained_weight, args.parts, args.continue_train)

    tester = Tester(0, configs)
    tester._make_batch_generator()

    # train
    for epoch in range(trainer.start_epoch, configs['end_epoch']):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            for key in targets.keys():
                targets[key] = targets[key].cuda()
            for key in meta_info.keys():
                meta_info[key] = meta_info[key].cuda()

            loss = trainer.model(inputs, targets, meta_info)
            loss = {k:loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, configs['end_epoch'], itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)

        # test
        tester._make_model_new(pretrained_weight, args.parts, epoch)
        eval_result = {}
        cur_sample_idx = 0
        for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
            # forward
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            for key in targets.keys():
                targets[key] = targets[key].cuda()
            for key in meta_info.keys():
                meta_info[key] = meta_info[key].cuda()
            with torch.no_grad():
                out = tester.model(inputs, targets, meta_info)

            # save output
            out = {k: v.cpu().numpy() for k, v in out.items()}
            for k, v in out.items(): batch_size = out[k].shape[0]
            out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

            # evaluate
            cur_eval_result = tester._evaluate(out, cur_sample_idx)
            for k, v in cur_eval_result.items():
                if k in eval_result:
                    eval_result[k] += v
                else:
                    eval_result[k] = v
            cur_sample_idx += len(out)

        tester._print_eval_result(eval_result)

if __name__ == "__main__":
    main()

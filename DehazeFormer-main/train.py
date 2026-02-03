import os
import sys
import argparse
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_fn

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *

# 可选：LPIPS 和 FID
try:
	import lpips
	HAS_LPIPS = True
except ImportError:
	HAS_LPIPS = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
try:
	from OHTER_MATHER.image_metrics import compute_fid
	HAS_FID = True
except ImportError:
	HAS_FID = False


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')
parser.add_argument('--data_ratio', default=1.0, type=float, help='使用数据的前 x (0-1)，如 0.1 表示 10%%')
parser.add_argument('--fid_freq', default=1, type=int, help='每 N 个 eval 轮次计算一次 FID，1=每轮都算')
parser.add_argument('--overwrite', action='store_true', help='已有模型时覆盖并重新训练')
parser.add_argument('--batch_size', default=None, type=int, help='覆盖配置中的 batch_size，显存不足时可设为 4 或 2')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()

	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(args.no_autocast):
			output = network(source_img)
			loss = criterion(output, target_img)

		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg


_LPIPS_FN = None

def valid(val_loader, network, epoch=0, compute_fid_this_epoch=False):
	"""验证：输出 PSNR、SSIM、LPIPS、FID"""
	global _LPIPS_FN
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	LPIPS_M = AverageMeter() if HAS_LPIPS else None
	pred_list, target_list = [], []  # 用于 FID

	torch.cuda.empty_cache()
	network.eval()

	if HAS_LPIPS and _LPIPS_FN is None:
		_LPIPS_FN = lpips.LPIPS(net='alex').cuda().eval()
	lpips_fn = _LPIPS_FN

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():
			output = network(source_img).clamp_(-1, 1)

		# [-1,1] -> [0,1]
		output_01 = output * 0.5 + 0.5
		target_01 = target_img * 0.5 + 0.5

		# PSNR
		eps = 1e-8
		mse = F.mse_loss(output_01, target_01, reduction='none').mean((1, 2, 3))
		psnr = (10 * torch.log10(1.0 / (mse + eps))).mean()
		PSNR.update(psnr.item(), source_img.size(0))

		# SSIM（与 test.py 一致，下采样）
		_, _, H, W = output_01.size()
		down_ratio = max(1, round(min(H, W) / 256))
		out_pool = F.adaptive_avg_pool2d(output_01, (int(H / down_ratio), int(W / down_ratio)))
		tgt_pool = F.adaptive_avg_pool2d(target_01, (int(H / down_ratio), int(W / down_ratio)))
		ssim_val = ssim_fn(out_pool, tgt_pool, data_range=1, size_average=True)
		SSIM.update(ssim_val.item(), source_img.size(0))

		# LPIPS（需 [0,1]）
		if lpips_fn is not None:
			with torch.no_grad():
				dist = lpips_fn(output_01 * 2 - 1, target_01 * 2 - 1)  # lpips 期望 [-1,1]
			LPIPS_M.update(dist.mean().item(), source_img.size(0))

		# 收集用于 FID
		if compute_fid_this_epoch and HAS_FID:
			# (B,C,H,W) -> list of (H,W,3) numpy [0,1]
			for i in range(output_01.size(0)):
				Pred = (output_01[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
				Targ = (target_01[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
				pred_list.append(Pred)
				target_list.append(Targ)

	metrics = {'psnr': PSNR.avg, 'ssim': SSIM.avg}
	if LPIPS_M is not None:
		metrics['lpips'] = LPIPS_M.avg
	else:
		metrics['lpips'] = None

	if compute_fid_this_epoch and HAS_FID and len(pred_list) > 0:
		try:
			fid_val = compute_fid(pred_list, target_list)
			metrics['fid'] = fid_val
		except Exception as e:
			metrics['fid'] = None
			metrics['fid_error'] = str(e)
	else:
		metrics['fid'] = None

	return metrics


if __name__ == '__main__':
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	network = eval(args.model.replace('-', '_'))()
	network = nn.DataParallel(network).cuda()

	criterion = nn.L1Loss()

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scaler = GradScaler()

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	batch_size = args.batch_size if args.batch_size is not None else setting['batch_size']
	train_dataset = PairLoader(dataset_dir, 'train', 'train', 
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'],
								data_ratio=args.data_ratio)
	train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], 
							  setting['patch_size'], data_ratio=args.data_ratio)
	val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)

	if args.data_ratio < 1.0:
		print(f'==> 使用数据前 {args.data_ratio*100:.0f}%: train={len(train_dataset)}, val={len(val_dataset)}')

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
		print('==> Start training, current model name: ' + args.model)
		if not HAS_LPIPS:
			print('    (LPIPS 未安装，跳过。pip install lpips)')
		if not HAS_FID:
			print('    (FID 未安装，跳过。需 OHTER_MATHER 模块: code/OHTER_MATHER)')
		# print(network)

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

		best_psnr = 0
		for epoch in tqdm(range(setting['epochs'] + 1)):
			loss = train(train_loader, network, criterion, optimizer, scaler)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()

			if epoch % setting['eval_freq'] == 0:
				compute_fid_this = (epoch // setting['eval_freq']) % args.fid_freq == 0
				metrics = valid(val_loader, network, epoch=epoch, compute_fid_this_epoch=compute_fid_this)

				avg_psnr = metrics['psnr']
				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				writer.add_scalar('valid_ssim', metrics['ssim'], epoch)
				if metrics['lpips'] is not None:
					writer.add_scalar('valid_lpips', metrics['lpips'], epoch)
				if metrics.get('fid') is not None:
					writer.add_scalar('valid_fid', metrics['fid'], epoch)

				# 控制台输出
				msg = f'Epoch {epoch} | PSNR: {avg_psnr:.2f} | SSIM: {metrics["ssim"]:.4f}'
				if metrics['lpips'] is not None:
					msg += f' | LPIPS: {metrics["lpips"]:.4f}'
				if metrics.get('fid') is not None:
					msg += f' | FID: {metrics["fid"]:.2f}'
				print(msg)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'.pth'))
				
				writer.add_scalar('best_psnr', best_psnr, epoch)

	else:
		print('==> Existing trained model')
		exit(1)

import os
import sys
import json
import argparse
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from my_dataset import PETDataset
from model.MFor import MADFormerModel
from utils import create_lr_scheduler, get_params_groups, train_one_epoch, evaluate

import warnings
warnings.filterwarnings('ignore')


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, lr_scheduler=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.)


def get_label_map(task_name):
    """Return label_map dict for a given task."""
    mappings = {
        'AD_HC':       {'AD': 1, 'HC': 0},
        'HC_MCI':      {'HC': 0, 'MCI': 1},
        'EMCI_LMCI':   {'EMCI': 0, 'LMCI': 1},
        'HC_ALL_MCI':  {'HC': 0, 'MCI': 1},
    }
    return mappings.get(task_name, None)


def main(args):
    setup_seed(3407)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    print(f"Dataset: {args.dataset} | Task: {args.task} | Fold: {args.fold}")

    # ---- Paths ----
    data_dir = args.data_dir     # e.g. "data/18F-AV1451"
    fold = args.fold

    train_csv = os.path.join(data_dir, args.task, f"train_fold{fold}.csv")
    val_csv   = os.path.join(data_dir, args.task, f"val_fold{fold}.csv")
    test_csv  = os.path.join(data_dir, args.task, f"test_fold{fold}.csv")

    print(f"Train CSV: {train_csv}")
    print(f"Val   CSV: {val_csv}")
    print(f"Test  CSV: {test_csv}")

    # ---- Output dirs ----
    exp_name = f"{args.dataset}_{args.task}_fold{fold}"
    weight_dir = os.path.join(args.output_dir, "weights", exp_name)
    result_dir = os.path.join(args.output_dir, "results")
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs", exp_name))

    # ---- Dataset ----
    label_map = get_label_map(args.task)
    if label_map is None:
        print(f"[WARNING] Unknown task '{args.task}', will auto-detect labels from CSV")

    batch_size = args.batch_size

    train_dataset = PETDataset(train_csv, data_dir, label_map=label_map, target_size=(128, 128, 128))
    val_dataset   = PETDataset(val_csv,   data_dir, label_map=label_map, target_size=(128, 128, 128))
    test_dataset  = PETDataset(test_csv,  data_dir, label_map=label_map, target_size=(128, 128, 128))

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers per process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, collate_fn=train_dataset.collate_fn, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, collate_fn=val_dataset.collate_fn, num_workers=nw)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, collate_fn=test_dataset.collate_fn, num_workers=nw)

    # ---- Model ----
    model = MADFormerModel(dataset='mf', _conv_repr=True, _pe_type="learned").to(device)

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    # ---- Resume ----
    checkpoint_path = os.path.join(weight_dir, "checkpoint.pth")
    best_model_path = os.path.join(weight_dir, "best_model.pth")
    result_json_path = os.path.join(result_dir, f"{exp_name}.json")
    start_epoch = 0
    best_acc = 0.
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"[Resume] Loading checkpoint from {checkpoint_path}")
        start_epoch, best_acc = load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler)
        # Also restore best_val_loss and patience_counter if saved
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        start_epoch += 1
        print(f"[Resume] Resuming from epoch {start_epoch}, best_acc={best_acc:.4f}, "
              f"best_val_loss={best_val_loss:.4f}, patience={patience_counter}/{args.patience}")
    elif os.path.exists(result_json_path):
        print(f"[Skip] Results already exist: {result_json_path}")
        tb_writer.close()
        return

    # ---- Training ----
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, _ = train_one_epoch(
            model=model, optimizer=optimizer, data_loader=train_loader,
            device=device, epoch=epoch, lr_scheduler=lr_scheduler)

        val_loss, val_acc, _ = evaluate(
            model=model, data_loader=val_loader, device=device, epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # Track best model (by val_acc)
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[Fold {fold}] Best model (acc={best_acc:.4f}) at epoch {epoch}")

        # Early stopping based on val_loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[Fold {fold}] Patience: {patience_counter}/{args.patience} "
                  f"(best_val_loss={best_val_loss:.4f}, current={val_loss:.4f})")

        if patience_counter >= args.patience:
            print(f"[Fold {fold}] Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

        # Save checkpoint for resume
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_acc': best_acc,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }, checkpoint_path)

    # ---- Test evaluation ----
    print(f"\n[Fold {fold}] Loading best model for test evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_metrics = evaluate(
        model=model, data_loader=test_loader, device=device, epoch=0)
    print(f"[Fold {fold}] Test -> loss: {test_loss:.4f}, acc: {test_acc:.4f}, best_val_acc: {best_acc:.4f}")

    # ---- Save results ----
    results = {
        'dataset': args.dataset,
        'task': args.task,
        'fold': fold,
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'best_val_acc': float(best_acc),
        **test_metrics,
    }
    with open(result_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_json_path}")

    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAD-Former Training')

    # Dataset
    parser.add_argument('--data-dir', type=str, default='data/18F-AV1451',
                        help='Base data directory')
    parser.add_argument('--dataset', type=str, default='18F-AV1451',
                        help='Dataset name (e.g. 18F-AV1451, 18F-AV45)')
    parser.add_argument('--task', type=str, default='AD_HC',
                        help='Classification task (AD_HC, HC_MCI, EMCI_LMCI, HC_ALL_MCI)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold (0-4) for 5-fold CV')

    # Output
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Root output directory for weights/results/logs')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience (epochs without val_loss improvement)')

    # Misc
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint if available')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Force training from scratch')
    parser.add_argument('--device', default='cuda:0',
                        help='Device (e.g. cuda:0, cpu)')

    opt = parser.parse_args()
    main(opt)

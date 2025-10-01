import os
import sys
import json
import time
import argparse
import logging
import hashlib
import pickle
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque

from operator import pos
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from torch.cuda.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import shutil


def _derive_no_eos_series(metrics: Dict[str, Any], split_prefix: str) -> Optional[Dict[str, list]]:
    """
    Try (1) direct series: <split>_loss_no_eos / <split>_ppl_no_eos
        else (2) derive from sums: <split>_{loss_sum, token_count, eos_loss_sum, eos_count}
        Return dict with keys 'steps', 'loss', 'ppl' or None if missing.
    """
    # 1) direct
    loss_key = f"{split_prefix}_loss_no_eos"
    ppl_key  = f"{split_prefix}_ppl_no_eos"
    if loss_key in metrics and ppl_key in metrics:
        steps = metrics.get("steps") or metrics.get(f"{split_prefix}_steps")
        return {"steps": steps, "loss": metrics[loss_key], "ppl": metrics[ppl_key]}

    # 2) derive from sums
    ls_key   = f"{split_prefix}_loss_sum"
    tc_key   = f"{split_prefix}_token_count"
    els_key  = f"{split_prefix}_eos_loss_sum"
    ec_key   = f"{split_prefix}_eos_count"

    if all(k in metrics for k in (ls_key, tc_key, els_key, ec_key)):
        loss_sum       = metrics[ls_key]
        token_count    = metrics[tc_key]
        eos_loss_sum   = metrics[els_key]
        eos_count      = metrics[ec_key]
        steps          = metrics.get("steps") or metrics.get(f"{split_prefix}_steps")

        # Align lengths defensively
        n = min(len(loss_sum), len(token_count), len(eos_loss_sum), len(eos_count))
        loss_no_eos = []
        ppl_no_eos  = []
        for i in range(n):
            tok_wo_eos = max(1, token_count[i] - eos_count[i])
            l_wo_eos = (loss_sum[i] - eos_loss_sum[i]) / tok_wo_eos
            loss_no_eos.append(l_wo_eos)
            ppl_no_eos.append(math.exp(l_wo_eos))
        if steps is not None:
            steps = steps[:n]
        return {"steps": steps, "loss": loss_no_eos, "ppl": ppl_no_eos}

    return None



# -----------------------------
# Complete Plotting System (4 files with all metrics)
# -----------------------------

class TrainingPlotter:
    """Dynamic plotting with aligned X-axes and zooming support."""

    def __init__(self, plots_dir: str, n_layers: int = 12, stm_capacity: Optional[int] = None):
        import os
        import matplotlib.pyplot as plt
        os.makedirs(plots_dir, exist_ok=True)
        self.plots_dir = plots_dir
        self.n_layers = n_layers
        self.stm_capacity = stm_capacity

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

        # Color gradient for layers (same style as before)
        self.gate_colors = []
        for i in range(n_layers):
            ratio = i / (n_layers - 1) if n_layers > 1 else 0.0
            if ratio < 0.33:
                local_ratio = ratio / 0.33
                r = 0; g = local_ratio; b = 1 - local_ratio
            elif ratio < 0.66:
                local_ratio = (ratio - 0.33) / 0.33
                r = local_ratio; g = 1 - local_ratio; b = 0
            else:
                local_ratio = (ratio - 0.66) / 0.34
                r = 1 - local_ratio * 0.5; g = 0; b = local_ratio * 0.8
            self.gate_colors.append((r, g, b))

    # ---------------- helpers ----------------
    @staticmethod
    def _smooth_ema(data, alpha=0.03):
        """Apply exponential moving average smoothing"""
        import numpy as np
        if len(data) == 0:
            return data
        smoothed = np.zeros_like(data, dtype=float)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed

    @staticmethod
    def _has(metrics: dict, *keys) -> bool:
        for k in keys:
            v = metrics.get(k, None)
            if v is None:
                return False
            if isinstance(v, (list, tuple)) and len(v) == 0:
                return False
        return True

    @staticmethod
    def _steps(metrics: dict, key: str):
        return metrics.get(key, [])

    @staticmethod
    def _align_events_to_ref(ref_steps, event_steps, event_values, fill_value=0.0, how="prev"):
        import numpy as np
        if not ref_steps:
            return []
        ref = np.asarray(ref_steps, dtype=np.int64)
        es  = np.asarray(event_steps, dtype=np.int64)
        ev  = np.asarray(event_values, dtype=np.float64)
        y   = np.full(len(ref), float(fill_value), dtype=np.float64)
        if es.size == 0:
            return y.tolist()

        idx = np.searchsorted(ref, es, side="right") - 1
        if how == "nearest":
            left  = np.clip(idx, 0, len(ref)-1)
            right = np.clip(idx+1, 0, len(ref)-1)
            use_r = np.abs(ref[right] - es) < np.abs(es - ref[left])
            idx   = np.where(use_r, right, left)

        idx = np.clip(idx, 0, len(ref)-1)
        for i, v in zip(idx, ev):
            y[int(i)] = float(v)
        return y.tolist()

    @staticmethod
    def _enhance(ax):
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(10))
        ax.yaxis.set_major_locator(MaxNLocator(8))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _best_index(values, mode: str):
        import numpy as np
        arr = np.asarray(values, dtype=float)
        mask = np.isfinite(arr)
        if not mask.any():
            return None
        arr = arr.copy()
        if mode == "min":
            arr[~mask] = np.inf
            return int(np.argmin(arr))
        elif mode == "max":
            arr[~mask] = -np.inf
            return int(np.argmax(arr))
        return None

    @staticmethod
    def _add_best_vline(ax, steps, values, mode: str, color: str, label_prefix: str):
        import numpy as np
        from matplotlib import transforms as mtransforms
        if not steps or not values:
            return
        idx = TrainingPlotter._best_index(values, mode)
        if idx is None or idx >= len(steps):
            return

        x_best = float(steps[idx])
        y_best = float(values[idx])

        # Draw the vline
        ax.axvline(x_best, color=color, linewidth=1.8, linestyle='-', alpha=0.95, zorder=5)

        # Put the text at the TOP of the axes, aligned to the vline in x
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            x_best, 0.98, f"{label_prefix} {y_best:.4g} @ {int(x_best)}",
            transform=trans,
            rotation=90, rotation_mode='anchor',
            va='top', ha='center',
            fontsize=8, color=color,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.6),
            zorder=6, clip_on=False
        )

    @staticmethod
    def _autoscale_y(ax, values, pad=0.05, min_range=0.02):
        import numpy as np
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < min_range:
            mid = 0.5 * (hi + lo)
            lo = mid - 0.5 * min_range
            hi = mid + 0.5 * min_range
        span = hi - lo
        ax.set_ylim(lo - pad*span, hi + pad*span)

    @staticmethod
    def _build_grid_spanned(spans, ncols=4, row_h=4.0, width=24.0):
        """Build a compact grid. Uses constrained_layout and fits the last row's actual width."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Pack spans into rows
        rows = []
        cur, left = [], ncols
        for sp in spans:
            sp = max(1, min(int(sp), ncols))
            if sp == ncols:
                if cur:
                    rows.append(cur); cur = []
                rows.append([ncols]); left = ncols
            else:
                if sp > left:
                    rows.append(cur); cur = []; left = ncols
                cur.append(sp); left -= sp
        if cur:
            rows.append(cur)

        nrows = max(1, len(rows))
        widest = max((sum(r) for r in rows), default=ncols)
        fig_w = width * (widest / ncols)
        fig_h = row_h * nrows

        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        gs = GridSpec(nrows, widest, figure=fig)

        positions = []
        for r, row in enumerate(rows):
            c = 0
            for sp in row:
                positions.append((r, slice(c, c + sp)))
                c += sp

        def ax_for(i):
            r, c_slice = positions[i]
            return fig.add_subplot(gs[r, c_slice])

        return fig, ax_for

    @staticmethod
    def _set_shared_xlim(ax, ref_steps):
        if ref_steps:
            ax.set_xlim(ref_steps[0], ref_steps[-1])

    @staticmethod
    def _align_series_to_ref(ref_steps, series_steps, series_values, fill_value=0.0):
        import numpy as np
        if not ref_steps:
            return []
        y = np.full(len(ref_steps), float(fill_value), dtype=float)
        if series_steps and series_values:
            m = {int(s): float(v) for s, v in zip(series_steps[:len(series_values)],
                                                  series_values[:len(series_steps)])}
            for i, s in enumerate(ref_steps):
                if int(s) in m:
                    y[i] = m[int(s)]
        return y.tolist()

    @staticmethod
    def _x_by_index(train_steps, series_len: int):
        return train_steps[:series_len]

    def _apply_zoom_to_figure(self, fig, axes_list, train_steps, epoch_window: int = 20000):
        """Zoom BOTH X and Y axes to focus on recent data"""
        if not train_steps or epoch_window <= 0:
            return
        import numpy as np

        current_step = train_steps[-1]
        x0 = max(train_steps[0], current_step - int(epoch_window))

        for ax in axes_list:
            # First set X-axis zoom
            ax.set_xlim(x0, current_step)

            # Then adjust Y-axis based on visible data
            y_smooth, y_all = [], []
            for line in ax.get_lines():
                xdata, ydata = line.get_data()
                if len(xdata) == 0:
                    continue
                xarr = np.asarray(xdata, dtype=float)
                yarr = np.asarray(ydata, dtype=float)
                mask = (xarr >= x0) & (xarr <= current_step) & np.isfinite(yarr)
                if not mask.any():
                    continue
                seg = yarr[mask]
                y_all.extend(seg)

                # Prioritize smoothed lines
                alpha = line.get_alpha()
                lw = line.get_linewidth()
                if (alpha is None or alpha >= 0.85) and lw >= 2.2:
                    y_smooth.extend(seg)

            use = y_smooth if y_smooth else y_all
            if use:
                # For log scale axes, handle differently
                if ax.get_yscale() == 'log':
                    yarr = np.asarray([y for y in use if y > 0], dtype=float)
                    if yarr.size > 0:
                        y_min, y_max = float(np.min(yarr)), float(np.max(yarr))
                        # Add padding in log space
                        log_pad = 0.1 * (np.log10(y_max) - np.log10(y_min))
                        y_min = 10 ** (np.log10(y_min) - log_pad)
                        y_max = 10 ** (np.log10(y_max) + log_pad)
                        ax.set_ylim(y_min, y_max)
                else:
                    yarr = np.asarray(use, dtype=float)
                    y_min, y_max = float(np.min(yarr)), float(np.max(yarr))
                    pad = max(1e-9, 0.1 * (y_max - y_min))
                    ax.set_ylim(y_min - pad, y_max + pad)

    def snapshot(self, epoch: int):
        import os, shutil
        bases = ["losses_accuracies", "memory_metrics", "srep_and_misc"]
        snap_dir = os.path.join(self.plots_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)

        for base in bases:
            # Copy normal version
            src = os.path.join(self.plots_dir, f"{base}.png")
            if os.path.exists(src):
                dst = os.path.join(snap_dir, f"{base}_epoch_{epoch}.png")
                shutil.copy2(src, dst)

            # Copy zoomed version if it exists
            src_zoomed = os.path.join(self.plots_dir, f"{base}_zoomed.png")
            if os.path.exists(src_zoomed):
                dst_zoomed = os.path.join(snap_dir, f"{base}_epoch_{epoch}_zoomed.png")
                shutil.copy2(src_zoomed, dst_zoomed)

    def plot_all(self, metrics: dict, step: int, epoch_window: Optional[int] = None):
        """Render all groups; if epoch_window is provided, use it for zoomed panels."""
        from typing import Optional
        win = int(epoch_window) if (epoch_window is not None) else 20000
        self.plot_losses_accuracies(metrics, step, zoomed=False, epoch_window=win)
        self.plot_losses_accuracies(metrics, step, zoomed=True,  epoch_window=win)
        self.plot_memory_metrics(metrics,  step, zoomed=False, epoch_window=win)
        self.plot_memory_metrics(metrics,  step, zoomed=True,  epoch_window=win)
        self.plot_srep_and_misc(metrics,   step, zoomed=False, epoch_window=win)
        self.plot_srep_and_misc(metrics,   step, zoomed=True,  epoch_window=win)

    def plot_losses_accuracies(self, metrics: dict, step: int, zoomed: bool = False, epoch_window: int = 20000):
        import os, numpy as np
        import matplotlib.pyplot as plt

        train_steps = self._steps(metrics, "train_steps")
        val_steps   = self._steps(metrics, "val_steps")

        panels = []
        spans  = []
        axes_list = []

        # Extend color palette to include alignment + totals
        colors = {
            "main": '#0077BE',
            "bow":  '#FFD93D',
            "adj":  '#FF6B35',
            "norm": '#5C7AEA',
            "align": '#2CA02C',
            "tot_train": '#8E44AD',
            "tot_val":   '#2ECC71',
        }

        train_no_eos = _derive_no_eos_series(metrics, "train")
        val_no_eos   = _derive_no_eos_series(metrics, "val")

        def plot_series(ax, steps, values, color, dashed=False, label=None, linewidth=1.8, alpha=1.0, marker=None):
            n = min(len(steps), len(values))
            if n > 0:
                markevery = max(1, n // 20) if marker else None
                ax.plot(steps[:n], values[:n],
                        linewidth=linewidth, linestyle='--' if dashed else '-',
                        color=color, alpha=alpha, label=label,
                        marker=marker, markersize=4 if marker else 0, markevery=markevery)

        def clamp_series(series_steps, values, fallback_steps):
            if not values:
                return [], []
            if series_steps:
                n = min(len(series_steps), len(values))
                return series_steps[:n], values[:n]
            n = min(len(fallback_steps), len(values))
            return fallback_steps[:n], values[:n]

        # ---------------- Row 1: Training phases ----------------
        if self._has(metrics, "phase", "train_steps"):
            def draw(ax):
                phases = metrics["phase"][:len(train_steps)]
                mapping = {"NoMem":0, "STM":1, "STM+LTM":2, "Full":3}
                y = [mapping.get(p,0) for p in phases]
                ax.plot(train_steps[:len(y)], y, color='#2E86AB', linewidth=2.5, drawstyle='steps-post')
                ax.fill_between(train_steps[:len(y)], 0, y, alpha=0.3, color='#2E86AB', step='post')
                ax.set_yticks([0,1,2,3]); ax.set_yticklabels(['NoMem','STM','STM+LTM','Full'])
                ax.set_xlabel('Step'); ax.set_ylabel('Phase'); ax.set_title('Training Phases', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # ---------------- Row 1: Next-token loss ----------------
        if self._has(metrics, "train_loss", "train_steps"):
            def draw(ax):
                tr = metrics["train_loss"][:len(train_steps)]
                plot_series(ax, train_steps, tr, colors["main"], dashed=False, label='Train')
                self._add_best_vline(ax, train_steps, tr, "min", colors["main"], "best train")
                if self._has(metrics, "val_loss", "val_steps"):
                    va = metrics["val_loss"]
                    plot_series(ax, val_steps, va, 'C1', dashed=True, label='Val')
                    self._add_best_vline(ax, val_steps, va, "min", 'C1', "best val")
                ax.set_xlabel('Step'); ax.set_ylabel('Loss (per token)')
                ax.set_title('Next-Token Prediction Loss', fontweight='bold')
                ax.legend(loc='upper right', framealpha=0.9)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if ((train_no_eos and train_no_eos.get("loss")) or
            (val_no_eos and val_no_eos.get("loss"))):
            def draw(ax):
                added = False
                if train_no_eos and train_no_eos.get("loss"):
                    steps_tr, vals_tr = clamp_series(train_no_eos.get("steps"), train_no_eos.get("loss"), train_steps)
                    if steps_tr:
                        plot_series(ax, steps_tr, vals_tr, colors["main"], dashed=False, label='Train (no EOS)')
                        self._add_best_vline(ax, steps_tr, vals_tr, "min", colors["main"], "best train no-eos")
                        added = True
                if val_no_eos and val_no_eos.get("loss"):
                    steps_val, vals_val = clamp_series(val_no_eos.get("steps"), val_no_eos.get("loss"), val_steps)
                    if steps_val:
                        plot_series(ax, steps_val, vals_val, 'C1', dashed=True, label='Val (no EOS)')
                        self._add_best_vline(ax, steps_val, vals_val, "min", 'C1', "best val no-eos")
                        added = True
                ax.set_xlabel('Step'); ax.set_ylabel('Loss (per token)')
                ax.set_title('Next-Token Loss (No EOS Tokens)', fontweight='bold')
                if added:
                    ax.legend(loc='upper right', framealpha=0.9)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # ---------------- Row 1: Token accuracy ----------------
        if self._has(metrics, "train_acc", "train_steps"):
            def draw(ax):
                tr = metrics["train_acc"][:len(train_steps)]
                plot_series(ax, train_steps, tr, 'C0', dashed=False, label='Train')
                self._add_best_vline(ax, train_steps, tr, "max", 'C0', "best train")
                if self._has(metrics, "val_acc", "val_steps"):
                    va = metrics["val_acc"]
                    plot_series(ax, val_steps, va, 'C1', dashed=True, label='Val')
                    self._add_best_vline(ax, val_steps, va, "max", 'C1', "best val")
                ax.set_xlabel('Step'); ax.set_ylabel('Accuracy')
                ax.set_title('Token Prediction Accuracy', fontweight='bold')
                ax.legend(loc='lower right', framealpha=0.9)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # ---------------- Row 1: Learning rate ----------------
        if self._has(metrics, "lr", "train_steps"):
            def draw(ax):
                y = metrics["lr"][:len(train_steps)]
                plot_series(ax, train_steps, y, '#C73E1D')
                ax.set_xlabel('Step'); ax.set_ylabel('Learning Rate')
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                ax.set_title('Learning Rate Schedule', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # ---------------- Row 2: Per-step aux losses ----------------
        if self._has(metrics, "bow_loss_per_step", "train_steps"):
            def draw(ax):
                vals = metrics["bow_loss_per_step"][:len(train_steps)]
                ax.plot(train_steps[:len(vals)], vals, color=colors["bow"], linewidth=1.6)
                self._add_best_vline(ax, train_steps, vals, "min", colors["bow"], "best")
                ax.set_xlabel('Step'); ax.set_ylabel('Loss')
                ax.set_title('Bag-of-Words Loss (per step)', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if self._has(metrics, "adjacency_loss_per_step", "train_steps"):
            def draw(ax):
                vals = metrics["adjacency_loss_per_step"][:len(train_steps)]
                ax.plot(train_steps[:len(vals)], vals, color=colors["adj"], linewidth=1.6)
                self._add_best_vline(ax, train_steps, vals, "min", colors["adj"], "best")
                ax.set_xlabel('Step'); ax.set_ylabel('Loss')
                ax.set_title('Adjacency Contrastive Loss (per step)', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if self._has(metrics, "alignment_loss_per_step", "train_steps"):
            def draw(ax):
                vals = metrics["alignment_loss_per_step"][:len(train_steps)]
                ax.plot(train_steps[:len(vals)], vals, color=colors["align"], linewidth=1.6)
                self._add_best_vline(ax, train_steps, vals, "min", colors["align"], "best")
                ax.set_xlabel('Step'); ax.set_ylabel('Loss')
                ax.set_title('Alignment Loss (per step)', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if self._has(metrics, "norm_reg_loss_per_step", "train_steps"):
            def draw(ax):
                vals = metrics["norm_reg_loss_per_step"][:len(train_steps)]
                ax.plot(train_steps[:len(vals)], vals, color=colors["norm"], linewidth=1.6)
                self._add_best_vline(ax, train_steps, vals, "min", colors["norm"], "best")
                ax.set_xlabel('Step'); ax.set_ylabel('Loss')
                ax.set_title('Norm Regularization Loss (per step)', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # ---------------- Row 3: Loss contributions (row-wide) ----------------
        if self._has(metrics, "main_loss_per_step", "train_steps"):
            def draw(ax):
                main = np.array(metrics["main_loss_per_step"][:len(train_steps)])
                bow  = np.array(metrics.get("bow_loss_per_step",       []))[:len(train_steps)]
                adj  = np.array(metrics.get("adjacency_loss_per_step", []))[:len(train_steps)]
                aln  = np.array(metrics.get("alignment_loss_per_step", []))[:len(train_steps)]
                norm = np.array(metrics.get("norm_reg_loss_per_step",  []))[:len(train_steps)]
                def pad(x):
                    return np.pad(x, (0, max(0, len(train_steps)-len(x))), mode='edge')[:len(train_steps)]
                main, bow, adj, aln, norm = pad(main), pad(bow), pad(adj), pad(aln), pad(norm)
                total = main + bow + adj + aln + norm + 1e-10
                ax.plot(train_steps, 100*main/total, label='Main', linewidth=2, color=colors["main"])
                if len(bow):  ax.plot(train_steps, 100*bow /total, label='BoW',  linewidth=2, color=colors["bow"])
                if len(adj):  ax.plot(train_steps, 100*adj /total, label='Adj',  linewidth=2, color=colors["adj"])
                if len(aln):  ax.plot(train_steps, 100*aln /total, label='Align',linewidth=2, color=colors["align"])
                if len(norm): ax.plot(train_steps, 100*norm/total, label='Norm', linewidth=2, color=colors["norm"])
                ax.set_ylim(0, 100); ax.legend(ncol=5, loc='upper right', framealpha=0.9)
                ax.set_xlabel('Step'); ax.set_ylabel('% of Total Loss')
                ax.set_title('Loss Component Contributions (Per-Step Basis)', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(4)

        # ---------------- Row 4: Standard LM Perplexity (Train vs Val) ----------------
        if self._has(metrics, "train_ppl_main_token", "train_steps") or self._has(metrics, "val_ppl_main_token", "val_steps"):
            def draw(ax):
                import numpy as np
                pos_vals = []

                if self._has(metrics, "train_ppl_main_token", "train_steps"):
                    tr = metrics["train_ppl_main_token"][:len(train_steps)]
                    if tr:
                        valid = [(s, v) for s, v in zip(train_steps[:len(tr)], tr) if np.isfinite(v) and v > 0]
                        if valid:
                            steps, vals = zip(*valid)
                            ax.plot(steps, vals, color=colors["tot_train"], linestyle='-', linewidth=2.2, label='Train PPL',
                                    marker='o', markersize=3, markevery=max(1, len(steps)//20))
                            self._add_best_vline(ax, steps, vals, "min", colors["tot_train"], "best train")
                            pos_vals.extend(vals)

                if self._has(metrics, "val_ppl_main_token", "val_steps"):
                    va = metrics["val_ppl_main_token"]
                    n = min(len(val_steps), len(va))
                    if n > 0:
                        valid = [(s, v) for s, v in zip(val_steps[:n], va[:n]) if np.isfinite(v) and v > 0]
                        if valid:
                            steps, vals = zip(*valid)
                            ax.plot(steps, vals, color=colors["tot_val"], linestyle='--', linewidth=2.2, label='Val PPL',
                                    marker='s', markersize=3, markevery=max(1, len(steps)//20))
                            self._add_best_vline(ax, steps, vals, "min", colors["tot_val"], "best val")
                            pos_vals.extend(vals)

                ax.set_yscale('log')
                if pos_vals:
                    lo = max(1e-3, min(pos_vals) * 0.9)
                    hi = max(pos_vals) * 1.1
                    if hi > lo:
                        ax.set_ylim(lo, hi)

                ax.set_xlabel('Step'); ax.set_ylabel('Perplexity (log)')
                ax.set_title('Token-level Perplexity (Main CE only)', fontweight='bold')
                ax.legend(loc='upper right', framealpha=0.9)
                ax.grid(True, which="both", ls="-", alpha=0.2)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)

            panels.append(draw); spans.append(1)


        if ((train_no_eos and train_no_eos.get("ppl")) or
            (val_no_eos and val_no_eos.get("ppl"))):
            def draw(ax):
                import numpy as np
                pos_vals = []

                if train_no_eos and train_no_eos.get("ppl"):
                    steps_tr, vals_tr = clamp_series(train_no_eos.get("steps"), train_no_eos.get("ppl"), train_steps)
                    if steps_tr:
                        valid = [(s, v) for s, v in zip(steps_tr, vals_tr) if np.isfinite(v) and v > 0]
                        if valid:
                            steps, vals = zip(*valid)
                            plot_series(ax, steps, vals, colors["tot_train"], dashed=False,
                                        label='Train PPL (no EOS)', linewidth=2.2, marker='o')
                            self._add_best_vline(ax, steps, vals, "min", colors["tot_train"], "best train no-eos")
                            pos_vals.extend(vals)

                if val_no_eos and val_no_eos.get("ppl"):
                    steps_val, vals_val = clamp_series(val_no_eos.get("steps"), val_no_eos.get("ppl"), val_steps)
                    if steps_val:
                        valid = [(s, v) for s, v in zip(steps_val, vals_val) if np.isfinite(v) and v > 0]
                        if valid:
                            steps, vals = zip(*valid)
                            plot_series(ax, steps, vals, colors["tot_val"], dashed=True,
                                        label='Val PPL (no EOS)', linewidth=2.2, marker='s')
                            self._add_best_vline(ax, steps, vals, "min", colors["tot_val"], "best val no-eos")
                            pos_vals.extend(vals)

                ax.set_yscale('log')
                if pos_vals:
                    lo = max(1e-3, min(pos_vals) * 0.9)
                    hi = max(pos_vals) * 1.1
                    if hi > lo:
                        ax.set_ylim(lo, hi)

                ax.set_xlabel('Step'); ax.set_ylabel('Perplexity (log)')
                ax.set_title('Token-level Perplexity (No EOS Tokens)', fontweight='bold')
                if pos_vals:
                    ax.legend(loc='upper right', framealpha=0.9)
                ax.grid(True, which="both", ls="-", alpha=0.2)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)

            panels.append(draw); spans.append(1)


        # ---------------- Row 4: Aux accuracies ----------------
        if self._has(metrics, "bow_accuracy", "train_steps"):
            def draw(ax):
                vals = metrics["bow_accuracy"][:len(train_steps)]
                n = min(len(vals), len(train_steps))
                if n > 0:
                    ax.plot(train_steps[:n], vals[:n], color='#73AB84', linewidth=1.6)
                    ax.axhspan(0.3, 0.5, alpha=0.08, color='#73AB84')
                    ax.axhline(y=0.4, color='#73AB84', linestyle=':', alpha=0.4)
                    self._add_best_vline(ax, train_steps[:n], vals[:n], "max", '#2E8B57', "best")
                ax.set_xlabel('Step'); ax.set_ylabel('Recall@K')
                ax.set_title('BoW Reconstruction Accuracy', fontweight='bold')
                ax.set_ylim(0, 1)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if self._has(metrics, "adjacency_accuracy", "train_steps"):
            def draw(ax):
                vals = metrics["adjacency_accuracy"][:len(train_steps)]
                n = min(len(vals), len(train_steps))
                if n > 0:
                    ax.plot(train_steps[:n], vals[:n], color=colors["adj"], linewidth=1.6)
                    ax.axhspan(0.5, 1.0, alpha=0.08, color='#73AB84')
                    ax.axhline(y=0.75, color='#73AB84', linestyle=':', alpha=0.4)
                    self._add_best_vline(ax, train_steps[:n], vals[:n], "max", colors["adj"], "best")
                ax.set_xlabel('Step'); ax.set_ylabel('Ranking Accuracy')
                ax.set_title('Adjacency Ranking Accuracy', fontweight='bold')
                ax.set_ylim(0, 1)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Alignment accuracy (ADDED)
        if self._has(metrics, "alignment_accuracy", "train_steps"):
            def draw(ax):
                vals = metrics["alignment_accuracy"][:len(train_steps)]
                n = min(len(vals), len(train_steps))
                if n > 0:
                    ax.plot(train_steps[:n], vals[:n], color=colors["align"], linewidth=1.6)
                    ax.axhspan(0.5, 0.8, alpha=0.08, color='#73AB84')
                    ax.axhline(y=0.65, color='#73AB84', linestyle=':', alpha=0.4)
                    self._add_best_vline(ax, train_steps[:n], vals[:n], "max", colors["align"], "best")
                ax.set_xlabel('Step'); ax.set_ylabel('Alignment Accuracy')
                ax.set_title('Token-S_REP Alignment Accuracy', fontweight='bold')
                ax.set_ylim(0, 1)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Row 5: weights and misc
        if self._has(metrics, "aux_weight_mult", "train_steps"):
            def draw(ax):
                mult = metrics["aux_weight_mult"][:len(train_steps)]
                ax.plot(train_steps[:len(mult)], mult[:len(train_steps)], color='#FF1493', linewidth=1.6)
                for y in [3, 2, 1.5, 1]:
                    ax.axhline(y=y, color=('r' if y==3 else 'orange' if y==2 else 'yellow' if y==1.5 else 'green'),
                            linestyle='--', alpha=0.25)
                ax.set_xlabel('Step'); ax.set_ylabel('Multiplier')
                ax.set_title('Auxiliary Loss Weight Multiplier', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if self._has(metrics, "memory_weight", "train_steps"):
            def draw(ax):
                weights = metrics["memory_weight"][:len(train_steps)]
                n = min(len(weights), len(train_steps))
                if n > 0:
                    ax.plot(train_steps[:n], weights[:n], color='#4ECDC4', linewidth=1.6)
                    ax.fill_between(train_steps[:n], 0, weights[:n], alpha=0.25, color='#4ECDC4')
                ax.set_xlabel('Step'); ax.set_ylabel('Weight'); ax.set_ylim(0, 1.1)
                ax.set_title('Memory Integration Weight', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Grad/speed/mem
        if self._has(metrics, "grad_norm"):
            def draw(ax):
                vals  = metrics["grad_norm"]
                steps = metrics.get("grad_norm_steps", [])
                if steps:
                    n = min(len(vals), len(steps))
                    ax.plot(steps[:n], vals[:n], color='#8B008B', linewidth=1.6)
                else:
                    # fallback
                    n = min(len(vals), len(train_steps))
                    ax.plot(train_steps[:n], vals[:n], color='#8B008B', linewidth=1.6)
                ax.set_xlabel('Step'); ax.set_ylabel('Gradient Norm')
                ax.set_title('Gradient Norm', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax); axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if self._has(metrics, "steps_per_second", "train_steps"):
            def draw(ax):
                vals = metrics["steps_per_second"][:len(train_steps)]
                ax.plot(train_steps[:len(vals)], vals, color='#228B22', linewidth=1.6)
                ax.set_xlabel('Step'); ax.set_ylabel('Steps/sec')
                ax.set_title('Training Speed', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if self._has(metrics, "gpu_memory_gb", "train_steps"):
            def draw(ax):
                vals = metrics["gpu_memory_gb"][:len(train_steps)]
                n = min(len(vals), len(train_steps))
                if n > 0:
                    ax.plot(train_steps[:n], vals[:n], color='#DC143C', linewidth=1.6)
                    ax.fill_between(train_steps[:n], 0, vals[:n], alpha=0.25, color='#DC143C')
                ax.set_xlabel('Step'); ax.set_ylabel('Memory (GB)')
                ax.set_title('GPU Memory Usage', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if not panels:
            return

        fig, ax_for = self._build_grid_spanned(spans, ncols=4, row_h=4.5, width=24.0)
        for i, draw in enumerate(panels):
            draw(ax_for(i))

        # Apply zooming if requested
        if zoomed:
            self._apply_zoom_to_figure(fig, axes_list, train_steps, epoch_window)
            plt.suptitle(f'Losses and Accuracies - Step {step:,} (Zoomed)', fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(self.plots_dir, 'losses_accuracies_zoomed.png'), dpi=150)
        else:
            plt.suptitle(f'Losses and Accuracies - Step {step:,}', fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(self.plots_dir, 'losses_accuracies.png'), dpi=150)
        plt.close()

    def plot_memory_metrics(self, metrics: dict, step: int, zoomed: bool = False, epoch_window: int = 20000):
        import os
        import matplotlib.pyplot as plt

        train_steps = self._steps(metrics, "train_steps")
        if not train_steps:
            return

        panels = []
        spans  = []
        axes_list = []

        # Memory Gates (row-wide)
        gate_keys = [
            k for k in metrics.keys()
            if k.startswith("gate_")
            and len(metrics[k]) > 0
            and k.split("_")[1].isdigit()       # <-- only 'gate_<layerindex>'
        ]
        if gate_keys:
            def draw(ax):
                for k in sorted(gate_keys, key=lambda x: int(x.split("_")[1])):
                    idx = int(k.split("_")[1])
                    vals = metrics[k]
                    n = min(len(vals), len(train_steps))
                    color = self.gate_colors[idx % len(self.gate_colors)]
                    ax.plot(
                        train_steps[:n], vals[:n],
                        label=f"L{idx}",
                        color=color, linewidth=1.5, alpha=0.9
                    )
                ax.legend(ncol=4, loc='upper right', fontsize=7, framealpha=0.9)
                ax.set_xlabel('Step'); ax.set_ylabel('Gate Value')
                ax.set_title('Memory Gates (All Layers)', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(4)

        # Self-Attention Gates (row-wide)
        self_gate_keys = [k for k in metrics.keys() if k.startswith("self_gate_") and len(metrics[k]) > 0]
        if self_gate_keys:
            def draw(ax):
                for k in sorted(self_gate_keys, key=lambda x: int(x.split("_")[2])):
                    vals = metrics[k]
                    n = min(len(vals), len(train_steps))
                    color = self.gate_colors[int(k.split("_")[2]) % len(self.gate_colors)]
                    ax.plot(train_steps[:n], vals[:n], label=f"L{int(k.split('_')[2])}",
                            color=color, linewidth=1.5, alpha=0.9)
                ax.legend(ncol=4, loc='upper right', fontsize=7, framealpha=0.9)
                ax.set_xlabel('Step'); ax.set_ylabel('Gate Value')
                ax.set_title('Self-Attention Gates (All Layers)', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(4)

        # LTM Size
        if self._has(metrics, "ltm_size"):
            def draw(ax):
                vals = metrics["ltm_size"]
                n = min(len(vals), len(train_steps))
                ax.plot(train_steps[:n], vals[:n], color='#1A535C', linewidth=2)
                ax.fill_between(train_steps[:n], 0, vals[:n], alpha=0.3, color='#1A535C')
                ax.set_xlabel('Step'); ax.set_ylabel('# Sentences')
                ax.set_title('Long-Term Memory Size', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Helper: draw any LTM similarity series aligned to train_steps with smoothing
        def sim_panel(title, key, color='#333', needs_smoothing=False):
            def draw(ax):
                vals = metrics.get(key, [])
                ltm_steps = metrics.get("ltm_steps", [])
                aligned = self._align_series_to_ref(train_steps, ltm_steps, vals, fill_value=0.0)
                if aligned:
                    if needs_smoothing:
                        # Raw data with lower opacity
                        ax.plot(train_steps[:len(aligned)], aligned, color=color, linewidth=2, alpha=0.5)
                        # Smoothed line
                        smoothed = self._smooth_ema(aligned)
                        ax.plot(train_steps[:len(aligned)], smoothed, color=color, linewidth=2.5)
                        self._autoscale_y(ax, smoothed)
                        self._add_best_vline(ax, train_steps, aligned, "max", color, "best")
                    else:
                        ax.plot(train_steps[:len(aligned)], aligned, color=color, linewidth=2)
                        self._autoscale_y(ax, aligned)
                        self._add_best_vline(ax, train_steps, aligned, "max", color, "best")
                ax.set_xlabel('Step'); ax.set_ylabel('Cosine Similarity')
                ax.set_title(title, fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            return draw

        if self._has(metrics, "ltm_first_sim"):
            panels.append(sim_panel("Top-1 Retrieval Similarity", "ltm_first_sim",  '#FF9F1C', needs_smoothing=True)); spans.append(1)
        if self._has(metrics, "ltm_mean_topk"):
            panels.append(sim_panel("Mean Top-K Retrieval Similarity", "ltm_mean_topk", '#2EC4B6', needs_smoothing=True)); spans.append(1)
        if self._has(metrics, "ltm_mean_valid"):
            panels.append(sim_panel("Mean Selected Items Similarity", "ltm_mean_valid", '#33658A', needs_smoothing=True)); spans.append(1)

        if self._has(metrics, "ltm_used_ratio"):
            def draw(ax):
                vals = metrics.get("ltm_used_ratio", [])
                ltm_steps = metrics.get("ltm_steps", [])
                aligned = self._align_series_to_ref(train_steps, ltm_steps, vals, fill_value=0.0)
                # Raw data with lower opacity
                ax.plot(train_steps[:len(aligned)], aligned, color='#6A4C93', linewidth=2, alpha=0.5)
                ax.fill_between(train_steps[:len(aligned)], 0, aligned, alpha=0.2, color='#6A4C93')
                # Smoothed line
                smoothed = self._smooth_ema(aligned, alpha=0.03)
                ax.plot(train_steps[:len(aligned)], smoothed, color='#6A4C93', linewidth=2.5)
                ax.set_xlabel('Step'); ax.set_ylabel('Ratio'); ax.set_ylim(0, 1)
                ax.set_title('LTM Used Ratio', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Query-mode comparison (aligned)
        if self._has(metrics, "query_tokens_first_sim") or self._has(metrics, "query_hybrid_first_sim"):
            def draw(ax):
                ltm_steps = metrics.get("ltm_steps", [])
                if self._has(metrics, "query_tokens_first_sim"):
                    y = self._align_series_to_ref(train_steps, ltm_steps, metrics["query_tokens_first_sim"], 0.0)
                    ax.plot(train_steps[:len(y)], y, label='Tokens-only', color='#0077BE', linewidth=2)
                if self._has(metrics, "query_hybrid_first_sim"):
                    y = self._align_series_to_ref(train_steps, ltm_steps, metrics["query_hybrid_first_sim"], 0.0)
                    ax.plot(train_steps[:len(y)], y, label='Hybrid', color='#FF6B6B', linewidth=2)
                ax.set_xlabel('Step'); ax.set_ylabel('Cosine Similarity')
                ax.set_title('Query Mode Comparison: Top-1', fontweight='bold')
                ax.legend(loc='lower right'); self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if self._has(metrics, "query_tokens_mean_topk") or self._has(metrics, "query_hybrid_mean_topk"):
            def draw(ax):
                ltm_steps = metrics.get("ltm_steps", [])
                if self._has(metrics, "query_tokens_mean_topk"):
                    y = self._align_series_to_ref(train_steps, ltm_steps, metrics["query_tokens_mean_topk"], 0.0)
                    ax.plot(train_steps[:len(y)], y, label='Tokens-only', color='#0077BE', linewidth=2)
                if self._has(metrics, "query_hybrid_mean_topk"):
                    y = self._align_series_to_ref(train_steps, ltm_steps, metrics["query_hybrid_mean_topk"], 0.0)
                    ax.plot(train_steps[:len(y)], y, label='Hybrid', color='#FF6B6B', linewidth=2)
                ax.set_xlabel('Step'); ax.set_ylabel('Cosine Similarity')
                ax.set_title('Query Mode Comparison: Mean Top-K', fontweight='bold')
                ax.legend(loc='lower right'); self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Distribution of actual STM sizes (not a time series; keep as histogram)
        if "actual_stm_sizes" in metrics and len(metrics["actual_stm_sizes"]) > 0:
            def draw(ax):
                import numpy as np
                stm_sizes = metrics["actual_stm_sizes"]
                if self.stm_capacity is not None:
                    cap = self.stm_capacity
                else:
                    cap = int(max(stm_sizes))
                cap = max(cap, int(max(stm_sizes)))
                ax.hist(stm_sizes, bins=range(0, int(cap)+2), alpha=0.7, color='#2E86AB', edgecolor='black')
                ax.axvline(x=int(cap), linestyle='--', label='Max Capacity')
                mean_size = np.mean(stm_sizes)
                ax.axvline(x=mean_size, color='g', linestyle='--', label=f'Mean: {mean_size:.1f}')
                ax.set_xlabel('STM Size'); ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Actual STM Sizes', fontweight='bold')
                ax.legend(); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)


        # LTM Replacement (aligned to global steps via ltm_replace_steps) with smoothing
        if self._has(metrics, "ltm_replace_sim"):
            def draw(ax):
                rep_vals  = metrics.get("ltm_replace_sim", [])
                rep_steps = metrics.get("ltm_replace_steps", [])
                aligned   = self._align_events_to_ref(train_steps, rep_steps, rep_vals, fill_value=0.0, how="prev")
                if aligned:
                    # Raw data with lower opacity
                    ax.plot(train_steps[:len(aligned)], aligned, color='#8B5A3C', linewidth=2, alpha=0.5)
                    # Smoothed line
                    smoothed = self._smooth_ema(aligned, alpha=0.03)
                    ax.plot(train_steps[:len(aligned)], smoothed, color='#8B5A3C', linewidth=2.5)
                    self._autoscale_y(ax, smoothed)
                # draw the vline using the TRUE event step, not the aligned index
                self._add_best_vline(ax, rep_steps, rep_vals, "max", '#8B5A3C', "best")
                ax.set_xlabel('Step'); ax.set_ylabel('Cosine Similarity')
                ax.set_title('LTM Replacement Similarity', fontweight='bold')
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        if not panels:
            return

        fig, ax_for = self._build_grid_spanned(spans, ncols=4, row_h=4.5, width=24.0)
        for i, draw in enumerate(panels):
            draw(ax_for(i))

        # Apply zooming if requested
        if zoomed:
            self._apply_zoom_to_figure(fig, axes_list, train_steps, epoch_window)
            plt.suptitle(f'Memory System Metrics - Step {step:,} (Zoomed)', fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(self.plots_dir, 'memory_metrics_zoomed.png'), dpi=150)
        else:
            plt.suptitle(f'Memory System Metrics - Step {step:,}', fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(self.plots_dir, 'memory_metrics.png'), dpi=150)
        plt.close()

    def plot_srep_and_misc(self, metrics: dict, step: int, zoomed: bool = False, epoch_window: int = 20000):
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        train_steps = self._steps(metrics, "train_steps")
        rep_steps   = metrics.get("ltm_replace_steps", train_steps)
        panels = []
        spans = []
        axes_list = []

        # Raw Norms (pre-normalization)
        if self._has(metrics, "srep_raw_norm_mean"):
            def draw(ax):
                raw_mean = metrics["srep_raw_norm_mean"][:len(train_steps)]
                raw_std  = metrics.get("srep_raw_norm_std", [0]*len(raw_mean))[:len(raw_mean)]
                n = min(len(raw_mean), len(train_steps))
                if n > 0:
                    ax.plot(train_steps[:n], raw_mean[:n], color='#8B4513', linewidth=2, label='Raw Mean')
                    lo = [m-s for m,s in zip(raw_mean[:n], raw_std[:n])]
                    hi = [m+s for m,s in zip(raw_mean[:n], raw_std[:n])]
                    ax.fill_between(train_steps[:n], lo, hi, alpha=0.3, color='#8B4513')
                    ax.axhspan(0.9, 1.1, alpha=0.1, color='#73AB84')
                    ax.axhline(y=1.0, color='#73AB84', linestyle='--', alpha=0.5)
                ax.set_xlabel('Step'); ax.set_ylabel('L2 Norm')
                ax.set_title('S_REP Raw Norms (Pre-normalization)', fontweight='bold')
                ax.legend(loc='upper right'); self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Diversity (sparse; align to its true steps if present)
        if self._has(metrics, "srep_diversity"):
            def draw(ax):
                import numpy as np
                vals = metrics["srep_diversity"]
                div_steps = metrics.get("srep_diversity_steps", [])
                if div_steps:
                    aligned = self._align_series_to_ref(train_steps, div_steps, vals, fill_value=np.nan)
                    arr = np.array(aligned, dtype=float)
                    mask = np.isfinite(arr)
                    xs = np.array(train_steps[:len(aligned)])[mask]
                    ys = arr[mask]
                    if ys.size:
                        ax.plot(xs, ys, color='#A23B72', linewidth=2)
                else:
                    # Back-compat (older logs with no step stamps)
                    n = min(len(vals), len(train_steps))
                    ax.plot(train_steps[:n], vals[:n], color='#A23B72', linewidth=2)

                ax.axhspan(0.3, 1.0, alpha=0.1, color='#73AB84')
                ax.axhspan(0.05, 0.3, alpha=0.1, color='#F18F01')
                ax.axhspan(0.0, 0.05, alpha=0.1, color='#C73E1D')
                ax.set_xlabel('Step'); ax.set_ylabel('Diversity (1 - mean cosine)')
                ax.set_title('S_REP Diversity', fontweight='bold')
                ax.set_ylim(0, 1); self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Max similarity (smoothed)
        if self._has(metrics, "srep_max_similarity"):
            def draw(ax):
                vals = metrics["srep_max_similarity"]; n = min(len(vals), len(train_steps))
                # Raw (faint) + EMA
                ax.plot(train_steps[:n], vals[:n], color='#FF6347', linewidth=2, alpha=0.5)
                smoothed = self._smooth_ema(vals[:n], alpha=0.03)
                ax.plot(train_steps[:n], smoothed, color='#FF6347', linewidth=2.5)
                ax.axhspan(0.0, 0.5, alpha=0.1, color='#73AB84')
                ax.axhspan(0.5, 0.8, alpha=0.1, color='#F18F01')
                ax.axhspan(0.8, 1.0, alpha=0.1, color='#C73E1D')
                ax.set_xlabel('Step'); ax.set_ylabel('Max Cosine Similarity')
                ax.set_title('S_REP Max Similarity', fontweight='bold')
                ax.set_ylim(0, 1); self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Adjacent coherence (smoothed)
        if self._has(metrics, "srep_adjacent_coherence"):
            def draw(ax):
                vals = metrics["srep_adjacent_coherence"]; n = min(len(vals), len(train_steps))
                ax.plot(train_steps[:n], vals[:n], color='#6C757D', linewidth=2, alpha=0.5)
                smoothed = self._smooth_ema(vals[:n], alpha=0.03)
                ax.plot(train_steps[:n], smoothed, color='#6C757D', linewidth=2.5)
                ax.axhspan(0.5, 0.8, alpha=0.1, color='#73AB84')
                ax.set_xlabel('Step'); ax.set_ylabel('Cosine Similarity')
                ax.set_title('Adjacent Sentence Coherence', fontweight='bold')
                ax.set_ylim(0, 1); self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Overall STM attention mass — keep only if present (aligned to attn_steps)
        if self._has(metrics, "attn_stm_mass"):
            steps_attn = metrics.get("attn_steps", [])
            def draw(ax):
                vals = metrics["attn_stm_mass"]
                aligned = self._align_series_to_ref(train_steps, steps_attn, vals, fill_value=0.0)
                if aligned:
                    ax.plot(train_steps[:len(aligned)], aligned, color='#4682B4', linewidth=2)
                    ax.fill_between(train_steps[:len(aligned)], 0, aligned, alpha=0.3, color='#4682B4')
                ax.set_xlabel('Step'); ax.set_ylabel('Attention Mass')
                ax.set_title('Overall Attention Mass on STM', fontweight='bold')
                ax.set_ylim(0, 1); self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # STM S_REP grad-norm (sparse; align and smooth)
        if self._has(metrics, "stm_srep_grad"):
            # helper for simple moving average
            def _ma(y, window=5):
                y = np.asarray(y, dtype=float)
                if window <= 1 or y.size < window:
                    return y
                kern = np.ones(int(window), dtype=float) / float(window)
                return np.convolve(y, kern, mode='valid')

            def draw(ax):
                vals = metrics["stm_srep_grad"]
                steps = metrics.get("stm_srep_grad_steps", [])
                if steps:
                    n = min(len(vals), len(steps))
                    xs = steps[:n]; ys = vals[:n]
                    ax.plot(xs, ys, color='#2F4F4F', linewidth=1.6, alpha=0.95, label='Raw')
                    ma = _ma(ys, window=5)
                    if ma.size > 0:
                        ax.plot(xs[-len(ma):], ma, color='#2F4F4F', linewidth=2.4, alpha=0.5, label='MA (w=5)')
                    self._autoscale_y(ax, ys)
                else:
                    x = self._x_by_index(train_steps, len(vals))
                    n = min(len(vals), len(x))
                    if n > 0:
                        ax.plot(x[:n], vals[:n], color='#2F4F4F', linewidth=1.6, alpha=0.95, label='Raw')
                        ma = _ma(vals[:n], window=5)
                        if ma.size > 0:
                            ax.plot(x[:n][-len(ma):], ma, color='#2F4F4F', linewidth=2.4, alpha=0.5, label='MA (w=5)')
                        self._autoscale_y(ax, vals[:n])
                ax.set_xlabel('Step'); ax.set_ylabel('Mean Grad Norm')
                ax.set_title('STM S_REP Grad-Norm (probed)', fontweight='bold')
                ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax); axes_list.append(ax)
            panels.append(draw); spans.append(1)

        # Component Gradient Norms (Cross/Self/FFN/LN), raw + EMA
        comp_map = [
            ("cross_attn",  "#1f77b4", "Cross-Attn"),
            ("self_attn",   "#ff7f0e", "Self-Attn"),
            ("ffn",         "#2ca02c", "FFN"),
            ("layernorm",   "#9467bd", "LayerNorm"),
        ]
        any_comp = any(self._has(metrics, f"comp_grad_{k}", f"comp_grad_{k}_steps") for k,_,_ in comp_map)
        if any_comp:
            def draw(ax):
                for key, color, label in comp_map:
                    vals = metrics.get(f"comp_grad_{key}", [])
                    steps_k = metrics.get(f"comp_grad_{key}_steps", [])
                    if not vals or not steps_k:
                        continue
                    aligned = self._align_series_to_ref(train_steps, steps_k, vals, fill_value=np.nan)
                    # raw trace
                    ax.plot(train_steps[:len(aligned)], aligned, color=color, linewidth=1.5, alpha=0.35, label=f"{label} (raw)")
                    # EMA on finite values
                    arr = np.array(aligned, dtype=float)
                    mask = np.isfinite(arr)
                    if mask.any():
                        ema = self._smooth_ema(arr[mask], alpha=0.03)
                        xs  = np.array(train_steps[:len(aligned)])[mask]
                        ax.plot(xs, ema, color=color, linewidth=2.5, label=label)
                ax.set_xlabel('Step'); ax.set_ylabel('Mean Grad Norm')
                ax.set_title('Block Component Gradient Norms', fontweight='bold')
                ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
                self._set_shared_xlim(ax, train_steps); self._enhance(ax)
                axes_list.append(ax)
            panels.append(draw); spans.append(2)

        # --- NEW: First & last block parameter grad norms ---
        last_idx = int(metrics.get("n_layers", self.n_layers)) - 1

        def draw_block_param_panel(ax, suffix: str, title: str):
            import numpy as np
            for key, color, label in comp_map:
                vals   = metrics.get(f"comp_grad_{key}_{suffix}", [])
                steps_ = metrics.get(f"comp_grad_{key}_{suffix}_steps", [])
                if not vals or not steps_:
                    continue
                aligned = self._align_series_to_ref(train_steps, steps_, vals, fill_value=np.nan)
                # raw trace
                ax.plot(train_steps[:len(aligned)], aligned, color=color, linewidth=1.5, alpha=0.35, label=f"{label} (raw)")
                # EMA
                arr = np.array(aligned, dtype=float)
                mask = np.isfinite(arr)
                if mask.any():
                    ema = self._smooth_ema(arr[mask], alpha=0.03)
                    xs  = np.array(train_steps[:len(aligned)])[mask]
                    ax.plot(xs, ema, color=color, linewidth=2.5, label=label)
            ax.set_xlabel('Step'); ax.set_ylabel('Param Grad Norm')
            ax.set_title(title, fontweight='bold')
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9, ncol=2)
            self._set_shared_xlim(ax, train_steps); self._enhance(ax)
            axes_list.append(ax)

        # Block 1 (L0)
        panels.append(lambda ax: draw_block_param_panel(ax, "L0", "Block 1 (L0) Component Grad Norms"))
        spans.append(2)

        # Block 11 (last, L{last_idx})
        panels.append(lambda ax, s=f"L{last_idx}": draw_block_param_panel(ax, s, f"Block {last_idx} Component Grad Norms"))
        spans.append(2)


        if not panels:
            return

        fig, ax_for = self._build_grid_spanned(spans, ncols=4, row_h=4.5, width=24.0)
        for i, draw in enumerate(panels):
            draw(ax_for(i))

        if zoomed:
            self._apply_zoom_to_figure(fig, axes_list, train_steps, epoch_window)
            plt.suptitle(f'S_REP Health and Miscellaneous - Step {step:,} (Zoomed)', fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(self.plots_dir, 'srep_and_misc_zoomed.png'), dpi=150)
        else:
            plt.suptitle(f'S_REP Health and Miscellaneous - Step {step:,}', fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(self.plots_dir, 'srep_and_misc.png'), dpi=150)
        plt.close()

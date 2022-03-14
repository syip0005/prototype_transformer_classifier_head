import torch
import argparse


def save_checkpoint(state: dict, filename: str = './model_param/save.pt'):

    """Save model state and parameters"""
    torch.save(state, filename)
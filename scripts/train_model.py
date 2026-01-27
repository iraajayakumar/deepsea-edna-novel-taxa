#!/usr/bin/env python3
"""
Main training script for IIC self-supervised clustering.
"""
import os
import sys
from pathlib import Path

# Fix paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.trainer import train

if __name__ == "__main__":
    print("🚀 Starting IIC Self-Supervised Training...")
    print(f"Project root: {PROJECT_ROOT}")
    train()

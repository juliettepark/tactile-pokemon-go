"""
Starts the tactile/handpose backend for pressure-triggered Pokemon capture.

Usage:
    python startBackend.py --label <label> --mode <mode>

Arguments:
    --label: Label for RECORD / SAVE_TO_CSV sessions (e.g. ball_blue)
    --mode: predict (default), record, or save_to_csv

Modes:

- predict: Stream pose continuously. When glove pressure crosses the capture
  threshold, classify the current hand pose as power_sphere or precision_pinch
  and send PREDICTION to the Quest.
- record: Wait for START/STOP from Unity, then collapse the take and append
  a labeled row to the result CSV.
- save_to_csv: Wait for START/STOP from Unity, then save the raw sequence CSV.
"""

import argparse

from flaskApp.index import start_server_web
from TouchSensorWireless import MultiProtocolReceiver
from dataToQuest import stream_to_quest
import streamHandposeAndGlove
from streamHandposeAndGlove import BackendMode, sync_quest_and_glove


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start the tactile/handpose backend (pressure-triggered Pokemon capture)."
    )
    parser.add_argument(
        "--label",
        type=str,
        default="none",
        help="Label for RECORD / SAVE_TO_CSV sessions (e.g. ball_blue)",
    )

    valid_modes = [m.value for m in BackendMode]
    parser.add_argument(
        "--mode",
        type=str,
        default=BackendMode.PREDICT.value,
        choices=valid_modes,
        help=f"Mode for the backend: {', '.join(valid_modes)}",
    )
    args = parser.parse_args()
    args.mode = BackendMode(args.mode)  # convert string to enum
    return args


args = parse_args()

# Set the label for the recording session in the streaming script (so we don't have to pass down)
streamHandposeAndGlove.current_session_label = args.label

# Set the mode for the backend
streamHandposeAndGlove.mode = args.mode

myReceiver = MultiProtocolReceiver(configFilePath='./configs/oneGloveSerialReceiverRightSmall.json')

# Listens for handpose from Quest, snapshots the glove, and in predict mode
# fires a grasp PREDICTION when pressure crosses the capture threshold.
myReceiver.runCustomMethod(sync_quest_and_glove)

# start_server_web(myReceiver)

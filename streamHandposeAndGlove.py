from enum import Enum
import sys
from pathlib import Path

# Ensure project root is on path so "utils" package resolves
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import atexit
import csv
import numpy as np
import pandas as pd
import asyncio
import websockets
import json
from datetime import datetime
import time
import joblib
from tactile_utils.PressureConverter import PressureConverter
from tactile_utils.DisplacementConverter import DisplacementConverter

from tactile_utils.tactile_handpose_utils import (
    NUM_BONE_VALUES,
    get_descriptive_headers,
    get_right_bone_headers,
    save_to_csv,
    save_to_result_data_csv,
)

# =============================== CONSTANTS ===============================

# Number of sensors in the tactile glove pressure grid
NUM_SENSORS = 16 * 16

# Raw glove ADC: lower value = more pressure. Unloaded cells sit near 3072.
UNLOADED_ADC = 3072.0

# A cell counts as pressed when its raw ADC is at or below this (lower = harder).
# One noisy cell around 350 should not fire; several cells must agree.
PRESSURE_THRESHOLD = 400.0
PRESSED_CELL_MIN = 6

# A cell still counts as contacting at or below this. Release when few remain.
RELEASE_CELL_ADC = 800.0
RELEASED_CELL_MAX = 1

# Consecutive 10ms frames required before press/release counts.
PRESS_DEBOUNCE_FRAMES = 5
RELEASE_DEBOUNCE_FRAMES = 8

# Minimum seconds between predictions.
COOLDOWN_S = 2.5

# WEBSOCKET_HOST = "10.0.0.78"
WEBSOCKET_HOST = "10.0.0.8"
WEBSOCKET_PORT = 8765

# Type of grasp to record (RECORD / SAVE_TO_CSV modes only)
GRASP_TYPE = "Heavy Wrap Internal"

RESULT_CSV = _project_root / "data" / "power_sphere" / "labeled_collapsed_results" / "hwi_labeled_results_collect_from_stream.csv"
RAW_DATA_RECORDINGS_FOLDER = _project_root / "data" / "pinch_dough"
CONVERTED_RECORDINGS_FOLDER = _project_root / "data" / "power_sphere" / "converted_data"

MODEL_FILE = _project_root / "grasp_classifier.joblib"

# TEMP: dump every live HAND_POSE from this backend (Pokemon app) so we can
# compare those frames to CSVs from the separate recording app.
DUMP_LIVE_FRAMES = False
# LIVE_DUMP_CSV = _project_root / "data" / "pokemon_pinch_data_2.csv"
LIVE_DUMP_CSV = _project_root / "data" / "pokemon_hwi_data_2.csv"


class BackendMode(Enum):
    PREDICT = "predict"
    RECORD = "record"
    SAVE_TO_CSV = "save_to_csv"

# =============================== GLOBAL VARIABLES ===============================

# Reference to the active Quest connection.
active_quest = None

# Buffer for sensor values from the tactile glove.
latest_sensor_values = []

# Last HAND_POSE `data` list from Unity (always overwritten).
latest_handpose = []

# Incoming HAND_POSE packets since the last status log (bring-up check).
pose_packets_since_log = 0

# Buffer to store the full recording data (timestamps, tactile, handpose)
recording_buffer = []

# Whether we are currently recording (RECORD / SAVE_TO_CSV modes).
is_recording = False

# If we are recording new data, label with this
# Argument from command line.
current_session_label = "none"

# Mode to run the backend in. Predict or record.
mode = BackendMode.PREDICT

# Model to make the prediction
model = None

# Initialize the pressure and displacement converters
pressure_converter = PressureConverter()
displacement_converter = DisplacementConverter(GRASP_TYPE)

live_dump_file = None
live_dump_writer = None
live_dump_rows = 0


def open_live_frame_dump():
    """Overwrite LIVE_DUMP_CSV and write the same headers as recording-app captures."""
    global live_dump_file, live_dump_writer, live_dump_rows
    if not DUMP_LIVE_FRAMES:
        return
    LIVE_DUMP_CSV.parent.mkdir(parents=True, exist_ok=True)
    live_dump_file = open(LIVE_DUMP_CSV, "w", newline="")
    live_dump_writer = csv.writer(live_dump_file)
    live_dump_writer.writerow(get_descriptive_headers())
    live_dump_file.flush()
    live_dump_rows = 0
    print(f"[dump] writing every HAND_POSE to {LIVE_DUMP_CSV}")


def close_live_frame_dump():
    global live_dump_file, live_dump_writer
    if live_dump_file is None:
        return
    live_dump_file.flush()
    live_dump_file.close()
    print(f"[dump] closed {LIVE_DUMP_CSV} ({live_dump_rows} frames)")
    live_dump_file = None
    live_dump_writer = None


def append_live_frame(unity_ts, bone_values):
    """Append one live pose frame (plus current glove snapshot) to the dump CSV."""
    global live_dump_rows
    if live_dump_writer is None:
        return
    if len(latest_sensor_values) == NUM_SENSORS:
        sensors = latest_sensor_values
    else:
        sensors = [0] * NUM_SENSORS
    bones = list(bone_values)
    if len(bones) < NUM_BONE_VALUES:
        bones.extend([0] * (NUM_BONE_VALUES - len(bones)))
    else:
        bones = bones[:NUM_BONE_VALUES]
    live_dump_writer.writerow(
        [datetime.now().timestamp(), unity_ts, *sensors, *bones]
    )
    live_dump_rows += 1
    if live_dump_rows % 50 == 0:
        live_dump_file.flush()


atexit.register(close_live_frame_dump)


def min_adc(sensor_values):
    """ADC of the most-pressed cell. Lower means more pressure."""
    if not sensor_values:
        return UNLOADED_ADC
    return float(np.min(np.asarray(sensor_values, dtype=float)))


def count_cells_at_or_below(sensor_values, threshold):
    """How many taxels are at or below an ADC threshold (i.e. pressed)."""
    if not sensor_values:
        return 0
    arr = np.asarray(sensor_values, dtype=float)
    valid = arr[arr > 0]
    return int(np.sum(valid <= threshold))


def predict_grasp(bone_values):
    """Classify a single right-hand pose as precision_pinch or heavy_wrap."""
    headers = get_right_bone_headers()
    offset = len(headers)
    right = list(bone_values[offset:offset + offset])
    X = pd.DataFrame([right], columns=headers)
    label = str(model.predict(X)[0])
    proba = ""
    if hasattr(model, "predict_proba"):
        p = dict(zip(model.classes_, model.predict_proba(X)[0]))
        proba = " " + str({k: round(float(v), 2) for k, v in p.items()})
    print(f"[grasp] {label}{proba}")
    return label


async def send_prediction(label):
    """Send a PREDICTION packet to the connected Quest, if any."""
    if active_quest is None:
        print("[⚠️] Pressure trigger fired but no Quest is connected")
        return
    # If prediciton successful, label as PREDICTION so Unity can detect and trigger callbacks
    payload = json.dumps({"type": "PREDICTION", "prediction": label})
    try:
        await active_quest.send(payload)
        print(f"[✅] PREDICTION {label} SENT TO UNITY")
    except websockets.ConnectionClosed:
        print("[⚠️] Quest disconnected while sending prediction")


async def quest_handler(websocket):
    """Handles incoming Hand Pose data from Unity."""
    global active_quest, latest_sensor_values, latest_handpose
    global recording_buffer, is_recording, current_session_label
    global pose_packets_since_log

    print(f"[🌐] Quest connected from {websocket.remote_address}")
    active_quest = websocket
    printed_first_pose = False

    try:
        async for message in websocket:
            try:
                payload = json.loads(message)
                msg_type = payload.get("type")

                if msg_type == "HAND_POSE":
                    data = payload.get("data") or []
                    # Overwrite the latest frame of handpose data
                    latest_handpose = data
                    pose_packets_since_log += 1
                    if not printed_first_pose:
                        print(f"[pose] first HAND_POSE len={len(latest_handpose)}")
                        printed_first_pose = True

                    if DUMP_LIVE_FRAMES:
                        append_live_frame(payload.get("ts"), data)

                    # If we don't want to predict and record instead, append a row
                    if mode != BackendMode.PREDICT and is_recording:
                        row = [
                            datetime.now().timestamp(),
                            payload.get("ts"),
                        ]
                        if len(latest_sensor_values) > 0:
                            row.extend(latest_sensor_values)
                        else:
                            row.extend([0] * NUM_SENSORS)
                        row.extend(data)
                        recording_buffer.append(row)

                elif msg_type == "START_RECORDING":
                    if mode == BackendMode.PREDICT:
                        continue
                    print("RECORDING STARTED")
                    recording_buffer = []
                    is_recording = True

                elif msg_type == "STOP_RECORDING":
                    if mode == BackendMode.PREDICT:
                        continue
                    is_recording = False

                    if mode == BackendMode.RECORD:
                        filename = f"tactile_hand_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}{current_session_label}.csv"
                        save_to_csv(recording_buffer, filename, RAW_DATA_RECORDINGS_FOLDER)
                        print(f"[✅] FULL CSV SAVED TO {RAW_DATA_RECORDINGS_FOLDER} as {filename}")
                        save_to_result_data_csv(
                            recording_buffer,
                            pressure_converter,
                            displacement_converter,
                            current_session_label,
                            CONVERTED_RECORDINGS_FOLDER,
                            RESULT_CSV,
                        )
                    elif mode == BackendMode.SAVE_TO_CSV:
                        filename = f"tactile_hand_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}{current_session_label}.csv"
                        save_to_csv(recording_buffer, filename, RAW_DATA_RECORDINGS_FOLDER)
                        print(f"[✅] RECORDING SAVED TO {filename}")

            except Exception as e:
                print(f"Error: {e}")
    finally:
        if active_quest is websocket:
            active_quest = None
        if live_dump_file is not None:
            live_dump_file.flush()
        print(f"[🌐] Quest disconnected (dump_frames={live_dump_rows})")


async def maybe_trigger_capture(
    n_pressed, n_contacting, was_pressed, last_fire_time, press_frames, release_frames
):
    """
    Debounced pressure trigger. Several cells must be pressed; one noisy
    cell should not fire or re-arm. Returns latch and debounce counters.
    """
    now = time.monotonic()

    if n_pressed >= PRESSED_CELL_MIN:
        press_frames += 1
    else:
        press_frames = 0

    if n_contacting <= RELEASED_CELL_MAX:
        release_frames += 1
    else:
        release_frames = 0

    if was_pressed:
        if release_frames >= RELEASE_DEBOUNCE_FRAMES:
            return False, last_fire_time, press_frames, release_frames
        return True, last_fire_time, press_frames, release_frames

    if press_frames < PRESS_DEBOUNCE_FRAMES:
        return False, last_fire_time, press_frames, release_frames
    if now - last_fire_time < COOLDOWN_S:
        return True, last_fire_time, press_frames, release_frames
    if not latest_handpose or len(latest_handpose) < NUM_BONE_VALUES:
        print(f"[⚠️] Pressure trigger (pressed_cells={n_pressed}) but no hand pose yet")
        return True, last_fire_time, press_frames, release_frames

    try:
        label = predict_grasp(latest_handpose)
    except Exception as e:
        print(f"[⚠️] Grasp prediction failed: {e}")
        return True, last_fire_time, press_frames, release_frames

    print(
        f"[🔴] CAPTURE pressed_cells={n_pressed} contacting={n_contacting} "
        f"pose_len={len(latest_handpose)} -> {label}"
    )
    await send_prediction(label)
    return True, now, 0, 0


async def sync_quest_and_glove(sensors):
    """
    The main background loop.
    1. Starts the WebSocket server to listen to the Quest.
    2. Continuously snapshots the tactile sensor grid from the ESP32 stream.
    3. Always keeps the latest hand pose. In predict mode, a rising-edge
       pressure trigger classifies that pose and sends PREDICTION to Unity.
    """
    global latest_sensor_values, model, pose_packets_since_log

    if MODEL_FILE.exists():
        model = joblib.load(MODEL_FILE)
        print(f"[🤖] Model loaded from {MODEL_FILE}")
    else:
        print(f"[⚠️] Grasp model not found at {MODEL_FILE}. "
              "Run processing_scripts/build_grasp_frame_dataset.py then models/grasp_classifier.py")

    print(f"[🚀] Sync Server Live on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    open_live_frame_dump()
    await websockets.serve(quest_handler, WEBSOCKET_HOST, WEBSOCKET_PORT)

    was_pressed = False
    last_fire_time = 0.0
    last_log_time = 0.0
    press_frames = 0
    release_frames = 0

    while True:
        lowest_adc = UNLOADED_ADC
        highest_adc = 0.0
        n_pressed = 0
        n_contacting = 0
        if sensors[0].init:
            try:
                latest_sensor_values = sensors[0].pressure.tolist()
                lowest_adc = min_adc(latest_sensor_values)
                highest_adc = float(np.max(latest_sensor_values)) if latest_sensor_values else 0.0
                n_pressed = count_cells_at_or_below(latest_sensor_values, PRESSURE_THRESHOLD)
                n_contacting = count_cells_at_or_below(latest_sensor_values, RELEASE_CELL_ADC)

                if mode == BackendMode.PREDICT:
                    was_pressed, last_fire_time, press_frames, release_frames = (
                        await maybe_trigger_capture(
                            n_pressed,
                            n_contacting,
                            was_pressed,
                            last_fire_time,
                            press_frames,
                            release_frames,
                        )
                    )
            except Exception as e:
                print(f"[glove] snapshot error: {e}")

        now = time.monotonic()
        if now - last_log_time >= 1.0:
            pose_n = len(latest_handpose) if latest_handpose else 0
            n_packets = pose_packets_since_log
            pose_packets_since_log = 0
            glove_ready = "yes" if sensors[0].init else "no"
            quest_ready = "yes" if active_quest else "no"
            print(
                f"[stream] HAND_POSE packets/s={n_packets} pose_len={pose_n} "
                f"min_adc={lowest_adc:.0f} max_adc={highest_adc:.0f} "
                f"pressed_cells={n_pressed} contacting={n_contacting} "
                f"glove={glove_ready} quest={quest_ready} "
                f"dump_frames={live_dump_rows}"
            )
            last_log_time = now

        await asyncio.sleep(0.01)

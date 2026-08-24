"""Utility functions for processing tactile handpose data from Unity."""

from datetime import datetime
import numpy as np
import pandas as pd
import sys
import csv
from pathlib import Path

from tactile_utils.PressureConverter import PressureConverter
from tactile_utils.DisplacementConverter import DisplacementConverter

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# =============================== CONSTANTS ===============================
# Number of sensors in the tactile glove pressure grid
NUM_SENSORS = 16*16

# File to save the results of the recording session. Labeled with 1 row per segment recording.
# RESULT_CSV = _project_root / "data" / "pinch_labeled_results_collect_from_stream.csv"
# RESULT_CSV = _project_root / "data" / "powergrasp_labeled_results_collect_from_stream.csv"

# RAW_DATA_RECORDINGS = _project_root / "data" / "raw_data"
# CONVERTED_RECORDINGS = _project_root / "data" / "converted_data"

# GRASP_TYPE = "Power sphere"
# GRASP_TYPE = "Precision pinch"
GRASP_TYPE = "Heavy Wrap Internal"

IMPORTANT_REGIONS = []
if GRASP_TYPE == "Power sphere":
    # Palm + Index + Middle + Ring + Little tips
    IMPORTANT_REGIONS = ["pa1", "pa2", "p3", "r3", "m3", "i3"]
elif GRASP_TYPE == "Precision pinch":
    IMPORTANT_REGIONS = ["i3", "t2"] # index finger and thumb
elif GRASP_TYPE == "Heavy Wrap Internal":
    # Palm (entire) 
    IMPORTANT_REGIONS = ["i3", "m3", "r3", "p3", "pa1", "pa2", "pa3", "pa4"]
    # IMPORTANT_REGIONS = ["i1", "i2", "m1", "m2", "r1", "r2", "p3", "pa3", "pa4", "pa1", "pa2"]

right_hand_regions = {
    't2':(slice(13,16), slice(12,16)),
    't1':(slice(11,13), slice(12,16)),
    'pa1':(slice(0,6), slice(12,16)),
    'pa2':(slice(6,11), slice(12,16)),
    'pa3':(slice(6,11), slice(9,12)),
    'pa4':(slice(0,6), slice(9,12)),
    'i1':(slice(9,11),slice(6,9)),
    'i2':(slice(9,11),slice(3,6)),
    'i3':(slice(9,11),slice(0,3)),
    'm1':(slice(6,8),slice(6,9)),
    'm2':(slice(6,8),slice(3,6)),
    'm3':(slice(6,8),slice(0,3)),
    'r1':(slice(3,5),slice(6,9)),
    'r2':(slice(3,5),slice(3,6)),
    'r3':(slice(3,5),slice(0,3)),
    'p1':(slice(0,2),slice(6,9)),
    'p2':(slice(0,2),slice(3,6)),
    'p3':(slice(0,2),slice(0,3)),
}

def get_region_from_index(i):
    # Get flattened index back into (row, col)
    row = i // 16
    col = i % 16

    # Check if the (row, col) is in any of the regions
    for name, (col_slice, row_slice) in right_hand_regions.items():
        if (col_slice.start <= col < col_slice.stop and 
            row_slice.start <= row < row_slice.stop):
            return name

    return "i3" # Default to index finger region if not found

# sensor index to region name
# Example:
"""
s_0: 'p3'
s_1: 'p3'
s_2: 'i3'
s_3: 'r3'
s_4: 'r3'
s_5: 'i3'
s_6: 'm3'
s_7: 'm3'
s_8: 'i3'
s_9: 'i3'
s_10: 'i3'
s_11: 'i3'
s_12: 'i3'
s_13: 'i3'
s_14: 'i3'
s_15: 'i3'
s_16: 'p3'
s_17: 'p3'
s_18: 'i3'
s_19: 'r3'
s_20: 'r3'
s_21: 'i3'
s_22: 'm3'
s_23: 'm3'
s_24: 'i3'
s_25: 'i3'
s_26: 'i3'
s_27: 'i3'
s_28: 'i3'
s_29: 'i3'
s_30: 'i3'
s_31: 'i3'
s_32: 'p3'
s_33: 'p3'
s_34: 'i3'
s_35: 'r3'
s_36: 'r3'
s_37: 'i3'
s_38: 'm3'
s_39: 'm3'
s_40: 'i3'
s_41: 'i3'
s_42: 'i3'
s_43: 'i3'
s_44: 'i3'
s_45: 'i3'
s_46: 'i3'
s_47: 'i3'
s_48: 'p2'
s_49: 'p2'
s_50: 'i3'
s_51: 'r2'
s_52: 'r2'
s_53: 'i3'
s_54: 'm2'
s_55: 'm2'
s_56: 'i3'
s_57: 'i2'
s_58: 'i2'
s_59: 'i3'
s_60: 'i3'
s_61: 'i3'
s_62: 'i3'
s_63: 'i3'
s_64: 'p2'
s_65: 'p2'
s_66: 'i3'
s_67: 'r2'
s_68: 'r2'
s_69: 'i3'
s_70: 'm2'
s_71: 'm2'
s_72: 'i3'
s_73: 'i2'
s_74: 'i2'
s_75: 'i3'
s_76: 'i3'
s_77: 'i3'
s_78: 'i3'
s_79: 'i3'
s_80: 'p2'
s_81: 'p2'
s_82: 'i3'
s_83: 'r2'
s_84: 'r2'
s_85: 'i3'
s_86: 'm2'
s_87: 'm2'
s_88: 'i3'
s_89: 'i2'
s_90: 'i2'
s_91: 'i3'
s_92: 'i3'
s_93: 'i3'
s_94: 'i3'
s_95: 'i3'
s_96: 'p1'
s_97: 'p1'
s_98: 'i3'
s_99: 'r1'
s_100: 'r1'
s_101: 'i3'
s_102: 'm1'
s_103: 'm1'
s_104: 'i3'
s_105: 'i1'
s_106: 'i1'
s_107: 'i3'
s_108: 'i3'
s_109: 'i3'
s_110: 'i3'
s_111: 'i3'
s_112: 'p1'
s_113: 'p1'
s_114: 'i3'
s_115: 'r1'
s_116: 'r1'
s_117: 'i3'
s_118: 'm1'
s_119: 'm1'
s_120: 'i3'
s_121: 'i1'
s_122: 'i1'
s_123: 'i3'
s_124: 'i3'
s_125: 'i3'
s_126: 'i3'
s_127: 'i3'
s_128: 'p1'
s_129: 'p1'
s_130: 'i3'
s_131: 'r1'
s_132: 'r1'
s_133: 'i3'
s_134: 'm1'
s_135: 'm1'
s_136: 'i3'
s_137: 'i1'
s_138: 'i1'
s_139: 'i3'
s_140: 'i3'
s_141: 'i3'
s_142: 'i3'
s_143: 'i3'
s_144: 'pa4'
s_145: 'pa4'
s_146: 'pa4'
s_147: 'pa4'
s_148: 'pa4'
s_149: 'pa4'
s_150: 'pa3'
s_151: 'pa3'
s_152: 'pa3'
s_153: 'pa3'
s_154: 'pa3'
s_155: 'i3'
s_156: 'i3'
s_157: 'i3'
s_158: 'i3'
s_159: 'i3'
s_160: 'pa4'
s_161: 'pa4'
s_162: 'pa4'
s_163: 'pa4'
s_164: 'pa4'
s_165: 'pa4'
s_166: 'pa3'
s_167: 'pa3'
s_168: 'pa3'
s_169: 'pa3'
s_170: 'pa3'
s_171: 'i3'
s_172: 'i3'
s_173: 'i3'
s_174: 'i3'
s_175: 'i3'
s_176: 'pa4'
s_177: 'pa4'
s_178: 'pa4'
s_179: 'pa4'
s_180: 'pa4'
s_181: 'pa4'
s_182: 'pa3'
s_183: 'pa3'
s_184: 'pa3'
s_185: 'pa3'
s_186: 'pa3'
s_187: 'i3'
s_188: 'i3'
s_189: 'i3'
s_190: 'i3'
s_191: 'i3'
s_192: 'pa1'
s_193: 'pa1'
s_194: 'pa1'
s_195: 'pa1'
s_196: 'pa1'
s_197: 'pa1'
s_198: 'pa2'
s_199: 'pa2'
s_200: 'pa2'
s_201: 'pa2'
s_202: 'pa2'
s_203: 't1'
s_204: 't1'
s_205: 't2'
s_206: 't2'
s_207: 't2'
s_208: 'pa1'
s_209: 'pa1'
s_210: 'pa1'
s_211: 'pa1'
s_212: 'pa1'
s_213: 'pa1'
s_214: 'pa2'
s_215: 'pa2'
s_216: 'pa2'
s_217: 'pa2'
s_218: 'pa2'
s_219: 't1'
s_220: 't1'
s_221: 't2'
s_222: 't2'
s_223: 't2'
s_224: 'pa1'
s_225: 'pa1'
s_226: 'pa1'
s_227: 'pa1'
s_228: 'pa1'
s_229: 'pa1'
s_230: 'pa2'
s_231: 'pa2'
s_232: 'pa2'
s_233: 'pa2'
s_234: 'pa2'
s_235: 't1'
s_236: 't1'
s_237: 't2'
s_238: 't2'
s_239: 't2'
s_240: 'pa1'
s_241: 'pa1'
s_242: 'pa1'
s_243: 'pa1'
s_244: 'pa1'
s_245: 'pa1'
s_246: 'pa2'
s_247: 'pa2'
s_248: 'pa2'
s_249: 'pa2'
s_250: 'pa2'
s_251: 't1'
s_252: 't1'
s_253: 't2'
s_254: 't2'
s_255: 't2'

"""
sensor_index_to_region = {f"s_{i}": get_region_from_index(i) for i in range(NUM_SENSORS*NUM_SENSORS)}

bone_names = [
    # Roots
    "XRHand_Wrist", "XRHand_Palm",

    # Thumb
    "XRHand_ThumbMetacarpal", "XRHand_ThumbProximal", "XRHand_ThumbDistal", "XRHand_ThumbTip",

    # Index
    "XRHand_IndexMetacarpal", "XRHand_IndexProximal", "XRHand_IndexIntermediate", "XRHand_IndexDistal", "XRHand_IndexTip",

    # Middle
    "XRHand_MiddleMetacarpal", "XRHand_MiddleProximal", "XRHand_MiddleIntermediate", "XRHand_MiddleDistal", "XRHand_MiddleTip",

    # Ring
    "XRHand_RingMetacarpal", "XRHand_RingProximal", "XRHand_RingIntermediate", "XRHand_RingDistal", "XRHand_RingTip",

    # Little
    "XRHand_LittleMetacarpal", "XRHand_LittleProximal", "XRHand_LittleIntermediate", "XRHand_LittleDistal", "XRHand_LittleTip"
]

# =============================== DATAFRAME CONVERSION ===============================

def get_sensor_data(df):
    """Get only the sensor data columns from the dataframe."""
    sensor_cols = [c for c in df.columns if c.startswith('s_')]
    return df[sensor_cols].values




# =============================== DATA HEADERS ===============================

def get_bone_headers():
    """
    Generates headers for the bone data.
    Ex.
    ["L_XRHand_Wrist_Px", "L_XRHand_Wrist_Py", "L_XRHand_Wrist_Pz", "L_XRHand_Wrist_Qx", "L_XRHand_Wrist_Qy", "L_XRHand_Wrist_Qz", "L_XRHand_Wrist_Qw",...]
    """
    headers = []
    for prefix in ["L", "R"]:
        for bone in bone_names:
            headers.append(f"{prefix}_{bone}_Px")
            headers.append(f"{prefix}_{bone}_Py")
            headers.append(f"{prefix}_{bone}_Pz")
            headers.append(f"{prefix}_{bone}_Qx")
            headers.append(f"{prefix}_{bone}_Qy")
            headers.append(f"{prefix}_{bone}_Qz")
            headers.append(f"{prefix}_{bone}_Qw")
    return headers


def get_right_bone_headers():
    """Right-hand columns from get_bone_headers() (R_XRHand_* Px/Py/Pz/Qx...)."""
    return [h for h in get_bone_headers() if h.startswith("R_")]


def get_descriptive_headers():
    """
    Generates headers matching both the sensor array and bone data for one segment recording.
    Ex.
    ["pc_ts", "unity_ts", "s_0", "s_1", ..., "s_255", "L_XRHand_Wrist_Px",...]
    """
    headers = ["pc_ts", "unity_ts"]
    
    # Create headers for the full sensor array
    # Format is s_0, s_1, ...
    for i in range(NUM_SENSORS):
        headers.append(f"s_{i}")
    
    # Create L_BoneName_Px, etc. for both hands
    headers.extend(get_bone_headers())
    # for prefix in ["L", "R"]:
    #     for bone in bone_names:
    #         headers.append(f"{prefix}_{bone}_Px")
    #         headers.append(f"{prefix}_{bone}_Py")
    #         headers.append(f"{prefix}_{bone}_Pz")
    #         headers.append(f"{prefix}_{bone}_Qx")
    #         headers.append(f"{prefix}_{bone}_Qy")
    #         headers.append(f"{prefix}_{bone}_Qz")
    #         headers.append(f"{prefix}_{bone}_Qw")
    return headers

def get_feature_headers():
    """
    Generates headers for the feature row of data.
    (1 recording collapsed into a single row of data for model training input)
    """
    # headers = ["index_avg", "index_min", "index_max", "thumb_avg", "thumb_min", "thumb_max", "finger_tip_avg", "finger_tip_min", "finger_tip_max", "label"]
    headers = []
    for region in IMPORTANT_REGIONS:
        headers.append(f"{region}_avg")
        headers.append(f"{region}_min")
        headers.append(f"{region}_max")
    
    headers.append("displacement_avg")
    headers.append("displacement_min")
    headers.append("displacement_max")

    headers.append("label")
    return headers

# def get_feature_headers():
#     """
#     Generates headers for the feature row of data.
#     (1 recording collapsed into a single row of data for model training input)
#     """
#     # headers = ["index_avg", "index_min", "index_max", "thumb_avg", "thumb_min", "thumb_max", "finger_tip_avg", "finger_tip_min", "finger_tip_max", "label"]
#     headers = []
#     for region in IMPORTANT_REGIONS:
#         headers.append(f"{region}_avg")
#         headers.append(f"{region}_min")
#         headers.append(f"{region}_max")
#         headers.append(f"{region}_std")
#         headers.append(f"{region}_time_to_peak")
#         headers.append(f"{region}_slope")

#     headers.append("displacement_avg")
#     headers.append("displacement_min")
#     headers.append("displacement_max")
#     headers.append("displacement_std")

#     headers.append("label")
#     return headers

# =============================== DATA PROCESSING/RECORDING ===============================

def recording_buffer_to_df(recording_buffer):
    """Convert recording_buffer (list of rows) to a DataFrame with descriptive headers."""
    if not recording_buffer:
        return None
    headers = get_descriptive_headers()
    return pd.DataFrame(recording_buffer, columns=headers)

def save_to_csv(recording_buffer, filename, location: Path):
    """Save the segment recording data into its own CSV file."""
    # Determine how many sensor columns we have based on the first row of data
    if not recording_buffer:
        return
    
    headers = get_descriptive_headers()
    
    with open(location / filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(recording_buffer)

def convert_raw_data_to_pressure(df: pd.DataFrame, pressure_converter: PressureConverter):
    """Convert the raw data into pressure values. MODIFIES THE DF"""
    for sensor_col in df.columns:
        if sensor_col.startswith('s_'):
            df[sensor_col] = pressure_converter.get_pressure(df[sensor_col], sensor_index_to_region[sensor_col])
    return df

def append_displacement_to_df(df: pd.DataFrame, displacement_converter: DisplacementConverter):
    """Append the displacement values to the DataFrame."""

    bone_cols = displacement_converter.bone_headers
    df['displacement_pc1'] = df[bone_cols].apply(
        lambda row: displacement_converter.get(row.tolist()), 
        axis=1
    )
    return df

def save_to_result_data_csv(recording_buffer, pressure_converter: PressureConverter, displacement_converter: DisplacementConverter, label: str, converted_recordings_folder: Path, result_csv: Path):
    """
    Save the segment recording data collapsed as a single row to the result CSV file WITH labels.
    Args:
        recording_buffer: List of rows of data (raw tactile and handpose data for one segment recording)
        pressure_converter: PressureConverter instance.
        displacement_converter: DisplacementConverter instance.
        label: Label for the recording.
    """

    # Convert list of data into DataFrame with descriptive headers
    df = recording_buffer_to_df(recording_buffer)
    if df is None:
        return

    # Transform ALL of sensor ADC values into pressure values and REPLACE in DataFrame
    df = convert_raw_data_to_pressure(df, pressure_converter)

    # Append the displacement values to the DataFrame
    df = append_displacement_to_df(df, displacement_converter)


    # # 2. Identify the bone headers 
    # # (Your class already pre-computes these as self.bone_headers)
    # bone_cols = displacement_converter.bone_headers

    # # 3. Calculate displacement for every frame (row) in the DataFrame
    # # axis=1 to process each row as a list
    # df['displacement_pc1'] = df[bone_cols].apply(
    #     lambda row: displacement_converter.get(row.tolist()), 
    #     axis=1
    # )

    # # UNCOMMENT TO SAVE THE CONVERTED FILE
    # # Define your filename
    # filename = f"processed_displacement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    # # Combine the directory path with the filename
    # output_path = converted_recordings_folder / filename
    # # Save the DataFrame
    # df.to_csv(output_path, index=False)
    # print(f"📁 Processed file saved to: {output_path}")


    # Collapse current recording into feature row
    data_row = collapse_recording_data(df)

    # Create result file with header if it doesn't exist yet
    file_exists = result_csv.exists()
    with open(result_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:

            # Write the headers for a collapsed row of data
            writer.writerow(get_feature_headers())

        # Write the collapsed row for the newest sequence recording with its label
        writer.writerow(data_row + [label])
        print(f"Wrote data row to {result_csv} with label {label}")

    # # also save for debugging
    # save_to_csv(f"tactile_hand_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

def collapse_recording_data(df: pd.DataFrame):
    """
    Collapse the recording data into a single row of data with averages, mins, and maxes pressures in each region
    and the average, min, and max of the distance between for the displacement vector.
    Expects DataFrame with pressure conversions and displacement conversions
    Returns a list of the data. NO LABEL
    """
    sensor_data = get_sensor_data(df)

    collapsed_row = []

    # Average pressure of the cells relevant to each region
    # Get 1 value per frame
    region_averages = []
    for region in IMPORTANT_REGIONS:
        region_averages.append(get_region_averages_right(sensor_data, region))

    # From these averaged values, get the average, min, max to collapse from list of frames into 1 value
    for average in region_averages:
        collapsed_row.append(np.mean(average))
        collapsed_row.append(np.min(average))
        collapsed_row.append(np.max(average))
    
    # Get the average, min, max of the displacement vector
    displacement = df['displacement_pc1']
    collapsed_row.append(np.mean(displacement))
    collapsed_row.append(np.min(displacement))
    collapsed_row.append(np.max(displacement))
    
    return collapsed_row

# def collapse_recording_data(df: pd.DataFrame):
#     """
#     Collapse the recording data into a single row of data with averages, mins, and maxes pressures in each region
#     and the average, min, and max of the distance between for the displacement vector.
#     Expects DataFrame with pressure conversions and displacement conversions
#     Returns a list of the data. NO LABEL
#     """
#     sensor_data = get_sensor_data(df)  # shape: (n_frames, 256)
#     n_frames = sensor_data.shape[0]  # number of frames

#     collapsed_row = []

#     # Use normalized time so slope features aren't directly dependent on recording length.
#     if n_frames <= 1:
#         t_norm = np.array([0.0])
#     else:
#         t_norm = np.linspace(0.0, 1.0, n_frames)

#     # --- Region features: mean/min/max/std + time_to_peak + slope ---
#     for region in IMPORTANT_REGIONS:
#         series = np.asarray(get_region_averages_right(sensor_data, region), dtype=float)

#         mean_v = float(np.mean(series)) if series.size else 0.0
#         min_v = float(np.min(series)) if series.size else 0.0
#         max_v = float(np.max(series)) if series.size else 0.0
#         std_v = float(np.std(series)) if series.size else 0.0

#         if series.size:
#             peak_idx = int(np.argmax(series))
#             time_to_peak = float(peak_idx / (series.size - 1)) if series.size > 1 else 0.0
#         else:
#             time_to_peak = 0.0

#         if series.size >= 2 and np.any(t_norm != t_norm[0]):
#             slope = float(np.polyfit(t_norm, series, 1)[0])
#         else:
#             slope = 0.0

#         collapsed_row.extend([mean_v, min_v, max_v, std_v, time_to_peak, slope])

#     # --- Displacement features ---
#     displacement = np.asarray(df["displacement_pc1"].values, dtype=float) if "displacement_pc1" in df.columns else np.asarray([], dtype=float)
#     d_mean = float(np.mean(displacement)) if displacement.size else 0.0
#     d_min = float(np.min(displacement)) if displacement.size else 0.0
#     d_max = float(np.max(displacement)) if displacement.size else 0.0
#     d_std = float(np.std(displacement)) if displacement.size else 0.0
#     collapsed_row.extend([d_mean, d_min, d_max, d_std])

#     return collapsed_row

# def collapse_recording_data(df: pd.DataFrame):
#     """
#     Collapse the recording data into a single row of data with averages, mins, and maxes of the index and thumb 
#     pressures and the average, min, and max of the distance between the index and thumb tips.
#     Returns a list of the data.
#     """
#     sensor_data = get_sensor_data(df)
#     index_pressures = get_index_averages_right(sensor_data)
#     thumb_pressures = get_thumb_averages_right(sensor_data)
#     index_avg = np.mean(index_pressures)
#     index_min = np.min(index_pressures)
#     index_max = np.max(index_pressures)
#     thumb_avg = np.mean(thumb_pressures)
#     thumb_min = np.min(thumb_pressures)
#     thumb_max = np.max(thumb_pressures)
#     finger_tip_distances = calculate_distance(df, "R_XRHand_IndexTip", "R_XRHand_ThumbTip")
#     finger_tip_avg = np.mean(finger_tip_distances)
#     finger_tip_min = np.min(finger_tip_distances)
#     finger_tip_max = np.max(finger_tip_distances)
    
#     return [index_avg, index_min, index_max, thumb_avg, thumb_min, thumb_max, finger_tip_avg, finger_tip_min, finger_tip_max]

# =============================== FEATURE EXTRACTION ===============================

def calculate_distance(df, bone1_prefix, bone2_prefix):
    """Calculates Euclidean distance between two bones."""
    p1 = df[[f"{bone1_prefix}_Px", f"{bone1_prefix}_Py", f"{bone1_prefix}_Pz"]].values
    p2 = df[[f"{bone2_prefix}_Px", f"{bone2_prefix}_Py", f"{bone2_prefix}_Pz"]].values
    return np.linalg.norm(p1 - p2, axis=1)

def get_region_averages_right(sensor_data, region_name):
    """Get the average pressure in the region for the right hand.
    Returns a list of average pressures for each frame.
    """
    region_pressure = []
    for frame in sensor_data:
        grid = frame.reshape(16, 16)
        region_pressure.append(np.mean(grid[right_hand_regions[region_name][1], right_hand_regions[region_name][0]]))
    return region_pressure

def get_index_averages_right(sensor_data):
    """Get the average pressure in the index finger region for the right hand.
    Returns a list of average pressures for each frame.
    """
    index_pressure = []
    for frame in sensor_data:
        grid = frame.reshape(16, 16)
        # BUG: These were flipped. Need rows 0:3 not cols 
        index_pressure.append(np.mean(grid[right_hand_regions['i3'][1], right_hand_regions['i3'][0]]))
    return index_pressure

def get_thumb_averages_right(sensor_data):
    """Get the average pressure in the thumb region for the right hand.
    Returns a list of average pressures for each frame.
    """
    thumb_pressure = []
    for frame in sensor_data:
        grid = frame.reshape(16, 16)
        thumb_pressure.append(np.mean(grid[right_hand_regions['t2'][1], right_hand_regions['t2'][0]]))
    return thumb_pressure


# Right-hand only. Knuckle-to-tip lengths plus palm-normal world direction.
# Palm normal points out of the palm (thumb side). In Quest tracking space,
# +Y is up; +X is right; +Z is forward. So -X is left (pinch) and +Z is away
# (power / back-of-hand).
GRASP_FEATURE_COLUMNS = [
    "dist_thumb_knuckle",
    "dist_index_knuckle",
    "dist_middle_knuckle",
    "dist_ring_knuckle",
    "dist_little_knuckle",
    "palm_nx",
    "palm_ny",
    "palm_nz",
    "palm_facing_left",
    "palm_facing_away",
]

_VALUES_PER_BONE = 7
_BONES_PER_HAND = len(bone_names)
NUM_BONE_VALUES = _BONES_PER_HAND * _VALUES_PER_BONE * 2  # L then R


def _xyz_from_bone_values(bone_values, hand="R"):
    """Map bone name -> (x, y, z) from the flattened Unity HAND_POSE data list."""
    hand_offset = 0 if hand == "L" else _BONES_PER_HAND * _VALUES_PER_BONE
    xyz = {}
    for i, bone in enumerate(bone_names):
        base = hand_offset + i * _VALUES_PER_BONE
        xyz[bone] = np.asarray(bone_values[base:base + 3], dtype=float)
    return xyz


def _xyz_from_named_row(row, hand="R"):
    """Map bone name -> (x, y, z) from a CSV row with R_XRHand_*_Px columns."""
    xyz = {}
    for bone in bone_names:
        xyz[bone] = np.array(
            [
                float(row[f"{hand}_{bone}_Px"]),
                float(row[f"{hand}_{bone}_Py"]),
                float(row[f"{hand}_{bone}_Pz"]),
            ],
            dtype=float,
        )
    return xyz


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _palm_basis(xyz):
    """Orthonormal palm frame: across (radial), along (fingers), normal (out of palm)."""
    wrist = xyz["XRHand_Wrist"]
    palm = xyz["XRHand_Palm"]
    index_mcp = xyz["XRHand_IndexMetacarpal"]
    little_mcp = xyz["XRHand_LittleMetacarpal"]
    thumb = xyz["XRHand_ThumbTip"]
    across = _unit(index_mcp - little_mcp)
    along = _unit(index_mcp - wrist)
    normal = np.cross(across, along)
    if np.dot(thumb - palm, normal) < 0:
        normal = -normal
    normal = _unit(normal)
    along = _unit(np.cross(normal, across))
    return palm, across, along, normal, np.linalg.norm(index_mcp - little_mcp)


def _grasp_features_from_xyz(xyz):
    """Right-hand knuckle-to-tip distances and palm facing in tracking space."""
    _palm, _across, _along, normal, scale = _palm_basis(xyz)
    if scale < 1e-5:
        raise ValueError("Hand scale too small to extract grasp features")

    pairs = [
        ("thumb", "XRHand_ThumbTip", "XRHand_ThumbMetacarpal"),
        ("index", "XRHand_IndexTip", "XRHand_IndexMetacarpal"),
        ("middle", "XRHand_MiddleTip", "XRHand_MiddleMetacarpal"),
        ("ring", "XRHand_RingTip", "XRHand_RingMetacarpal"),
        ("little", "XRHand_LittleTip", "XRHand_LittleMetacarpal"),
    ]
    feats = {}
    for name, tip, knuckle in pairs:
        feats[f"dist_{name}_knuckle"] = float(np.linalg.norm(xyz[tip] - xyz[knuckle]))
    feats["palm_nx"] = float(normal[0])
    feats["palm_ny"] = float(normal[1])
    feats["palm_nz"] = float(normal[2])
    # Tracking space: -X is left, +Z is away from a user facing +Z.
    feats["palm_facing_left"] = float(-normal[0])
    feats["palm_facing_away"] = float(normal[2])
    return [feats[col] for col in GRASP_FEATURE_COLUMNS]


def extract_grasp_features(bone_values, hand="R"):
    """
    Extract pose-only grasp features from a Unity HAND_POSE `data` list.
    `bone_values` is L then R, each bone Px,Py,Pz,Qx,Qy,Qz,Qw (see get_bone_headers).
    Returns a list in GRASP_FEATURE_COLUMNS order.
    """
    if bone_values is None or len(bone_values) < NUM_BONE_VALUES:
        raise ValueError(
            f"Expected at least {NUM_BONE_VALUES} bone values, got "
            f"{0 if bone_values is None else len(bone_values)}"
        )
    return _grasp_features_from_xyz(_xyz_from_bone_values(bone_values, hand="R"))


def extract_grasp_features_from_row(row, hand="R"):
    """Extract pose-only grasp features from a named CSV / Series row."""
    return _grasp_features_from_xyz(_xyz_from_named_row(row, hand="R"))


def right_hand_is_tracked(row):
    """True if the right-hand wrist/palm are not an all-zero missing pose."""
    try:
        wrist = np.array(
            [float(row["R_XRHand_Wrist_Px"]), float(row["R_XRHand_Wrist_Py"]), float(row["R_XRHand_Wrist_Pz"])]
        )
        palm = np.array(
            [float(row["R_XRHand_Palm_Px"]), float(row["R_XRHand_Palm_Py"]), float(row["R_XRHand_Palm_Pz"])]
        )
    except (KeyError, TypeError, ValueError):
        return False
    return not (np.allclose(wrist, 0) and np.allclose(palm, 0))


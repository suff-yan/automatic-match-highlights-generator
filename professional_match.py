import numpy as np
import librosa
from moviepy import VideoFileClip, concatenate_videoclips
from scipy.ndimage import gaussian_filter1d

# ===============================
# CONFIG
# ===============================

VIDEO_PATH = "input/match.mp4"
OUTPUT_PATH = "output/highlights.mp4"

WINDOW_SIZE = 20          # seconds per candidate window
PRE_ROLL = 5              # seconds before spike
POST_ROLL = 10            # seconds after spike
MAX_HIGHLIGHT_DURATION = 600  # 10 minutes (600 sec)
HOP_LENGTH = 512

# ===============================
# STEP 1: LOAD AUDIO
# ===============================

print("Loading audio...")
y, sr = librosa.load(VIDEO_PATH, sr=None)

# ===============================
# STEP 2: COMPUTE RMS ENERGY
# ===============================

print("Computing RMS energy...")
rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

# Convert frame index to time
times = librosa.frames_to_time(
    np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH
)

# Smooth the signal
rms_smooth = gaussian_filter1d(rms, sigma=3)

# Normalize using z-score
rms_norm = (rms_smooth - np.mean(rms_smooth)) / np.std(rms_smooth)

# ===============================
# STEP 3: WINDOW SCORING
# ===============================

print("Scoring windows...")

video_duration = times[-1]
num_windows = int(video_duration // WINDOW_SIZE)

window_scores = []

for i in range(num_windows):
    start_time = i * WINDOW_SIZE
    end_time = start_time + WINDOW_SIZE
    
    mask = (times >= start_time) & (times < end_time)
    if np.sum(mask) == 0:
        continue
    
    score = np.mean(rms_norm[mask])
    window_scores.append((start_time, end_time, score))

# Sort windows by excitement score (descending)
window_scores.sort(key=lambda x: x[2], reverse=True)

# ===============================
# STEP 4: SELECT TOP WINDOWS
# ===============================

print("Selecting top windows...")

selected_segments = []
total_duration = 0

for start, end, score in window_scores:
    segment_start = max(0, start - PRE_ROLL)
    segment_end = min(video_duration, end + POST_ROLL)
    segment_duration = segment_end - segment_start
    
    if total_duration + segment_duration <= MAX_HIGHLIGHT_DURATION:
        selected_segments.append((segment_start, segment_end))
        total_duration += segment_duration
    
    if total_duration >= MAX_HIGHLIGHT_DURATION:
        break

# ===============================
# STEP 5: MERGE OVERLAPPING SEGMENTS
# ===============================

print("Merging overlapping segments...")

selected_segments.sort(key=lambda x: x[0])
merged = []

for seg in selected_segments:
    if not merged:
        merged.append(seg)
    else:
        prev = merged[-1]
        if seg[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], seg[1]))
        else:
            merged.append(seg)

# Trim if slightly over 10 minutes
final_segments = []
current_total = 0

for start, end in merged:
    seg_len = end - start
    if current_total + seg_len <= MAX_HIGHLIGHT_DURATION:
        final_segments.append((start, end))
        current_total += seg_len
    else:
        remaining = MAX_HIGHLIGHT_DURATION - current_total
        if remaining > 0:
            final_segments.append((start, start + remaining))
        break

print(f"Final highlight duration: {current_total:.2f} seconds")

# ===============================
# STEP 6: GENERATE HIGHLIGHT VIDEO
# ===============================

print("Rendering highlight video...")

video = VideoFileClip(VIDEO_PATH)

clips = []
for start, end in final_segments:
    clips.append(video.subclipped(start, end))

final_video = concatenate_videoclips(clips)
final_video.write_videofile(OUTPUT_PATH)
print("Done. Highlights saved.")
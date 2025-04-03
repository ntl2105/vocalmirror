import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import librosa.display
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import re
import json
import io
from datetime import datetime

# --- Utility Functions ---

def is_nang_tone(syllable):
    """
    Check if the syllable has the T4 (nặng) tone marker (a dot below).
    """
    return "̣" in syllable

def split_into_syllables(text):
    """
    Split text into syllables, handling punctuation and common typos.
    Ensures there's a space after each punctuation mark.
    """
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Add space after punctuation if there isn't one
    text = re.sub(r'([.,!?;:])', r'\1 ', text)
    
    # Split on spaces
    syllables = text.split()
    
    # Filter out empty strings
    syllables = [s for s in syllables if s]
    
    return syllables

def estimate_syllable_boundaries(y, sr, syllables, hop_length=256):
    """
    Estimate syllable boundaries using a combination of onset detection and energy features.
    """
    # Get onset frames with a smaller hop length for finer detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Apply smoothing to reduce noise
    onset_env = gaussian_filter1d(onset_env, sigma=2)
    
    # Find peaks in the onset envelope
    peaks, _ = find_peaks(onset_env, distance=int(0.1 * sr / hop_length))  # Minimum 100ms between peaks
    
    # Convert frames to samples
    onset_samples = librosa.frames_to_samples(peaks, hop_length=hop_length)
    
    # If we have more onsets than syllables, keep only the strongest ones
    if len(onset_samples) > len(syllables):
        # Use onset strength to select the strongest onsets
        onset_strengths = onset_env[peaks]
        strongest_indices = np.argsort(onset_strengths)[-len(syllables):]
        onset_samples = onset_samples[strongest_indices]
    
    # Sort onsets
    onset_samples = np.sort(onset_samples)
    
    # Add start and end points
    boundaries = np.concatenate([[0], onset_samples, [len(y)]])
    
    # If we have fewer boundaries than needed, interpolate
    if len(boundaries) < len(syllables) + 1:
        boundaries = np.linspace(0, len(y), len(syllables) + 1, dtype=int)
    
    return boundaries

@st.cache_data
def load_audio(audio_path):
    """
    Load and cache audio file.
    """
    return librosa.load(audio_path, sr=None)

def plot_syllables_segmented(y, sr, syllables, boundaries, window_size=8.0, annotations=None):
    """
    Plot the audio waveform in chunks with syllable boundaries and labels.
    Each chunk shows a portion of the waveform with its corresponding syllables.
    """
    # Calculate time points for the waveform
    time = np.linspace(0, len(y) / sr, len(y))
    
    # Initialize annotations if not provided
    if annotations is None:
        annotations = {}
    
    # Calculate number of chunks
    total_duration = len(y) / sr
    num_chunks = int(np.ceil(total_duration / window_size))
    
    # Add summary section at the top
    st.subheader("Segmentation Summary")
    if len(boundaries) - 1 != len(syllables):
        st.warning(f"⚠️ Mismatch: {len(boundaries) - 1} segments but {len(syllables)} syllables")
    else:
        st.success(f"✓ Matched: {len(syllables)} syllables and segments")
    
    # Create a figure for each chunk
    for chunk_idx in range(num_chunks):
        start_time = chunk_idx * window_size
        end_time = min((chunk_idx + 1) * window_size, total_duration)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Create the figure for this chunk
        fig = go.Figure()
        
        # Add waveform for this chunk
        chunk_time = time[start_sample:end_sample] - start_time  # Adjust time to start at 0 for each chunk
        fig.add_trace(go.Scatter(
            x=chunk_time,
            y=y[start_sample:end_sample],
            mode='lines',
            name='Waveform',
            line=dict(color='blue', width=1),
            opacity=0.6
        ))
        
        # Add syllable boundaries and labels for this chunk
        for i in range(len(boundaries) - 1):
            b_start = boundaries[i]
            b_end = boundaries[i + 1]
            
            # Only show boundaries that overlap with this chunk
            if b_start <= end_sample and b_end >= start_sample:
                # Calculate relative position within chunk
                rel_start = max(0, b_start - start_sample)
                rel_end = min(end_sample - start_sample, b_end - start_sample)
                
                # Add boundary line (adjusted to chunk time)
                b_sec = (b_start - start_sample) / sr
                fig.add_vline(x=b_sec, line_dash="dash", line_color="green", opacity=0.5)
                
                # Add syllable label
                mid = (b_start + b_end) // 2
                if start_sample <= mid <= end_sample:
                    mid_sec = (mid - start_sample) / sr
                    syll = syllables[i] if i < len(syllables) else ""
                    color = "red" if is_nang_tone(syll) else "black"
                    
                    # Create hover text with boundary limits
                    hover_text = (
                        f"{syll}<br>Start: {b_start / sr:.2f} s<br>"
                        f"End: {b_end / sr:.2f} s"
                    )
                    
                    # Add label with background for better visibility
                    fig.add_annotation(
                        x=mid_sec,
                        y=0.4,
                        text=syll,
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-30,
                        font=dict(color=color, size=12),
                        hovertext=hover_text,
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=2
                    )
        
        # Add final boundary if it's in this chunk
        if len(boundaries) > 0 and boundaries[-1] <= end_sample and boundaries[-1] >= start_sample:
            fig.add_vline(x=(boundaries[-1] - start_sample) / sr, line_dash="dash", line_color="green", opacity=0.5)
        
        # Update layout with correct x-axis range
        fig.update_layout(
            title=f"Chunk {chunk_idx + 1}: {start_time:.2f}s – {end_time:.2f}s",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(range=[0, min(window_size, end_time - start_time)])  # Set x-axis range for each chunk
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the audio chunk
        chunk_audio = y[start_sample:end_sample]
        st.audio(chunk_audio, sample_rate=sr)
        
        # Add boundary adjustment controls for this chunk
        st.subheader(f"Boundary Adjustment (Chunk {chunk_idx + 1})")
        
        # Get syllables in this chunk
        chunk_syllables = []
        for i in range(len(boundaries) - 1):
            mid = (boundaries[i] + boundaries[i + 1]) // 2
            if start_sample <= mid <= end_sample:
                syll = syllables[i] if i < len(syllables) else ""
                chunk_syllables.append((i, syll))
        
        # Create columns for boundary controls
        num_cols = min(3, len(chunk_syllables))
        cols = st.columns(num_cols)
        
        for idx, (i, syll) in enumerate(chunk_syllables):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.write(f"Boundary {i+1} ({syll}):")
                
                # Add syllable audio playback
                syll_audio = y[boundaries[i]:boundaries[i+1]]
                st.audio(syll_audio, sample_rate=sr)
                
                # Add boundary adjustment buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("←", key=f"left_{i}_{chunk_idx}"):
                        if i > 0:
                            boundaries[i] = max(boundaries[i] - int(0.1 * sr), boundaries[i-1])
                            st.success(f"Moved left boundary for '{syll}'")
                with col2:
                    if st.button("→", key=f"right_{i}_{chunk_idx}"):
                        if i < len(boundaries) - 2:
                            boundaries[i+1] = min(boundaries[i+1] + int(0.1 * sr), boundaries[i+2])
                            st.success(f"Moved right boundary for '{syll}'")
        
        st.markdown("---")  # Add separator between chunks
    
    # Add export button at the end
    if st.button("Export Segmentation"):
        export_data = {
            "syllables": syllables,
            "boundaries": boundaries.tolist(),
            "annotations": annotations,
            "timestamp": datetime.now().isoformat(),
            "audio_file": os.path.basename(audio_path)
        }
        
        # Convert to JSON
        json_str = json.dumps(export_data, indent=2)
        
        # Create download button
        st.download_button(
            label="Download Segmentation",
            data=json_str,
            file_name=f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# --- Streamlit App Configuration and Main Code ---

st.set_page_config(layout="wide")
st.title("Vietnamese Syllable Segment Visualizer")

@st.cache_data
def load_metadata():
    return pd.read_csv("applied_tool/data/vimd_subset_metadata.csv")

# Load metadata
metadata = load_metadata()

# Create selection widgets
col1, col2 = st.columns(2)
with col1:
    province_list = sorted(metadata["province"].unique())
    default_province = "CaoBang" if "CaoBang" in province_list else province_list[0]
    selected_province = st.selectbox("Select a province", province_list, index=province_list.index(default_province))

with col2:
    # Filter speakers for selected province
    province_speakers = sorted(metadata[metadata["province"] == selected_province]["speaker"].unique())
    default_speaker = province_speakers[0] if len(province_speakers) > 0 else None
    selected_speaker = st.selectbox("Select a speaker", province_speakers, index=0 if default_speaker else None)

# Filter metadata for the selected province and speaker
province_files = metadata[
    (metadata["province"] == selected_province) & 
    (metadata["speaker"] == selected_speaker)
]

if not province_files.empty:
    selected_row = province_files.sample(1).iloc[0]
    audio_path = os.path.join("applied_tool/data/audio", selected_row["filename"])
    transcript = selected_row["text"]

    # Display transcript
    st.markdown(f"**Transcript:** {transcript}")

    # Load audio file (using cached function)
    y, sr = load_audio(audio_path)

    # Process transcript to extract syllables and estimate boundaries
    syllables = split_into_syllables(transcript)
    boundaries = estimate_syllable_boundaries(y, sr, syllables)
    
    # Initialize session state for annotations if not exists
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}

    # Display the full audio at the top
    st.subheader("Full Audio")
    st.audio(y, sample_rate=sr)
    
    # Add window size slider
    window_size = st.slider("Chunk Size (seconds)", min_value=2.0, max_value=8.0, value=8.0, step=0.5)

    # Plot the segmented syllable visualization with annotations
    plot_syllables_segmented(y, sr, syllables, boundaries, window_size=window_size, annotations=st.session_state.annotations)
    
    # Add feedback form
    st.subheader("Segmentation Feedback")
    feedback = st.text_area("Please provide feedback on the segmentation quality:", height=100)
    if st.button("Submit Feedback"):
        if feedback:
            # Save feedback to a file
            feedback_data = {
                "audio_file": os.path.basename(audio_path),
                "transcript": transcript,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            }
            
            # Create feedback directory if it doesn't exist
            os.makedirs("applied_tool/data/feedback", exist_ok=True)
            
            # Save feedback to file
            feedback_file = f"applied_tool/data/feedback/feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(feedback_file, "w") as f:
                json.dump(feedback_data, f, indent=2)
            
            st.success("Thank you for your feedback!")
else:
    st.write("No files available for the selected province and speaker.")
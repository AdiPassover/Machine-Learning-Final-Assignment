import os
import numpy as np
import librosa


class FSDDLoader:
    """
    Loader and feature extractor for the Free Spoken Digit Dataset (FSDD).
    Supports three modes:
      - 'aggregate': mean+std over time (compact fixed vector, classical ML)
      - 'flatten': fixed-length flattened MFCC sequence (classical ML, time ignored)
      - 'sequence': full MFCC time sequence (NNs, time preserved)
    """

    def __init__(self,
                 dataset_path="recordings",
                 feature_type="mfcc",     # "mfcc" or "mel"
                 n_mfcc=13,               # Number of MFCC coefficients
                 sample_rate=8000,        # Target sample rate (Hz)
                 frame_length=0.025,      # Frame length in seconds
                 frame_stride=0.010,      # Frame step in seconds
                 mode="aggregate",        # "aggregate", "flatten", or "sequence"
                 max_duration=None,       # Crop/pad to this length in seconds
                 fixed_frames=None        # For 'flatten' or 'sequence': number of frames to pad/trim to
                 ):
        """
        Parameters
        ----------
        dataset_path : str
            Path to the folder containing .wav files.
        feature_type : str
            "mfcc" or "mel".
        n_mfcc : int
            Number of MFCC coefficients (only for MFCC mode).
        sample_rate : int
            Resample audio to this rate.
        frame_length : float
            Analysis window length in seconds (default 25 ms).
        frame_stride : float
            Step between successive frames in seconds (default 10 ms).
        mode : str
            "aggregate" (mean+std), "flatten" (fixed vector), or "sequence" (T × F).
        max_duration : float or None
            Crop/pad raw audio to this max length in seconds before feature extraction.
        fixed_frames : int or None
            If given, pad/trim features to exactly this many frames (used in 'flatten'/'sequence').
        """
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.mode = mode
        self.max_duration = max_duration
        self.fixed_frames = fixed_frames

        assert self.mode in ("aggregate", "flatten", "sequence")
        assert self.feature_type in ("mfcc", "mel")
        assert self.max_duration is None or self.fixed_frames is None

    def _extract_features(self, audio):
        """Extract MFCC or mel-spectrogram features from an audio signal."""
        hop_length = int(self.frame_stride * self.sample_rate)  # jump between frames
        win_length = int(self.frame_length * self.sample_rate)  # window length

        if self.feature_type == "mfcc":
            mfccs = librosa.feature.mfcc(y=audio,
                                         sr=self.sample_rate,
                                         n_mfcc=self.n_mfcc,
                                         n_fft=win_length,
                                         hop_length=hop_length)
            return mfccs  # shape: (n_mfcc, T)
        elif self.feature_type == "mel":
            mel_spec = librosa.feature.melspectrogram(y=audio,
                                                      sr=self.sample_rate,
                                                      n_fft=win_length,
                                                      hop_length=hop_length)
            return librosa.power_to_db(mel_spec, ref=np.max)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def _pad_or_trim_frames(self, feat_matrix):
        """Pad or trim along time axis to fixed number of frames."""
        if self.fixed_frames is None:
            return feat_matrix
        T = feat_matrix.shape[1]  # number of time frames
        if T > self.fixed_frames:  # trim
            return feat_matrix[:, :self.fixed_frames]
        elif T < self.fixed_frames:  # pad
            pad_width = self.fixed_frames - T
            return np.pad(feat_matrix, ((0, 0), (0, pad_width)), mode='constant')
        return feat_matrix

    def load_data(self):
        """
        Loads the dataset and returns (X, y, speakers).
        """
        X, y, speakers = [], [], []
        files = [f for f in os.listdir(self.dataset_path) if f.endswith(".wav")]

        for filename in files:
            parts = filename.split("_")
            digit = int(parts[0])
            speaker = parts[1]
            file_path = os.path.join(self.dataset_path, filename)

            audio, _ = librosa.load(file_path, sr=self.sample_rate)

            # Optional crop/pad audio length
            if self.max_duration:
                target_len = int(self.max_duration * self.sample_rate)
                if len(audio) > target_len:
                    audio = audio[:target_len]
                else:
                    audio = np.pad(audio, (0, target_len - len(audio)))

            features = self._extract_features(audio)  # (F, T)

            if self.mode == "aggregate":
                mean_features = np.mean(features, axis=1)
                std_features = np.std(features, axis=1)
                features = np.concatenate([mean_features, std_features])
                X.append(features)

            elif self.mode == "flatten":
                features = self._pad_or_trim_frames(features)  # still (F, T_fixed)
                features = features.flatten()  # (F × T_fixed,)
                X.append(features)

            elif self.mode == "sequence":
                features = self._pad_or_trim_frames(features)  # (F, T_fixed) or (F, T_var)
                X.append(features.T)  # (T, F) for NN input

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            y.append(digit)
            speakers.append(speaker)

        # Convert to array where possible
        if self.mode in ("aggregate", "flatten"):
            X = np.array(X)

        return X, np.array(y), np.array(speakers)

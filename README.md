The main dataset used in this project and used by us to compare teams performances is DAPS
(Device and Produced Speech) Dataset (https://zenodo.org/records/4660670 ).
Six speakers from this dataset: F1, F7, F8, M3, M6, M8, form Class 1. The other 14 speakers belong to Class 0.

As a final user:

1. I am initializing the listening procedure.
2. The program listens to me with the use of a laptop microphone and detects the class.
3. Detected class and, optionally, confidence level, is displayed on the screen. The Jupyter-notebook program is used in this scenario.

The chosen metrics for evaluation - f1-score.
The chosen models: Neural - CNN, MLP, Classical - Random Forest Classifier together with RandomizedSearchCV applied.

! Preliminary analysis:
i. In the **main** we're generating random spectograms for raw wav files and processed data:
Here are the key observations:

Observations made:

1. Raw data spectograms (RDS) are over 2 min long
2. RDS contain excess information - not clear patterns, noises (particularly at lower frequencies)
3. RDS have a denser frequency range, likely due to background noises, irrelevant data (like silence) - we cannot really focus on essential audio features

Thus, preprocessing techniques were applied to each audio file like (trimming silences, reducing noises and cutting up to 10 sec)

As a result we have processed spectograms (it's visible in spectograms samples (also 5)):

1. Processed spectrograms are much shorter (10 seconds) compared to raw ones.
   This trimming helps focus on essential audio features and removes excess information.
2. Noise Reduction: Processed spectrograms appear cleaner, with clearer patterns, indicating effective noise reduction and silence trimming.
3. Frequency Information: Processed spectrograms highlight more distinct frequency patterns relevant to the voice signal.
4. Target Information: Processed spectrograms better capture the essential voice characteristics, making them more suitable for training a model focused on voice recognition.

In summary, processed spectrograms provide a clearer, noise-reduced representation of voice signals, which should enhance model performance by focusing on relevant features.

ii. Histograms
The histograms were created to explore the characteristics of the spectrograms generated from the audio data.

Mean Intensity: This represents the average amplitude of the spectrogram across time and frequency.
Variance in Intensity: This indicates how much the intensity varies.

We calculated the mean and variance for each spectrogram generated from the raw and processed audio files.
Separate histograms were generated for both the mean and variance values. These were created for both the raw and processed spectrograms, allowing us to compare the distributions.

Observations:
Mean intensity:
Raw Spectrograms: The mean intensity of raw spectrograms was lower and more spread out, indicating that raw audio files generally had lower average amplitude.
Processed Spectrograms: The processed spectrograms showed a more concentrated distribution with slightly higher mean intensities. This is likely due to noise reduction and normalization, which raised the average amplitude by removing background noise.

Distribution of Variance in Intensity:
Raw Spectrograms: The raw data had a wider spread in variance, suggesting that there was more variability in the audio’s intensity levels, possibly due to background noise or uneven recording conditions.
Processed Spectrograms: The processed spectrograms had a more concentrated variance distribution, often lower than the raw data. This suggests that noise reduction and other processing steps made the audio more uniform and reduced extreme intensity changes.

Result:
The histograms suggest that processing steps (like noise reduction and trimming) made the spectrograms more consistent by increasing mean intensity and reducing variance. This could be beneficial for our CNN model, as it reduces unnecessary variations and focuses on essential audio features.

Main CNN model:
The chosen activation function - Relu
Splitting - 70-20-10
Optimizer - Adam
Scheduler - StepLR (step_size=5, gamma=0.1)
Noise Reduction - noisereduse Fast Fourier transform (FFT)

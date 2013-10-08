## Time-Frequency Audio Segmentation Tool Set

Version 0.1: Extremely buggy, work in progress

TFASTS is a set of command-line tools for manipulating audio in the spectrogram domain.

You can use TFASTS to remove some noise from audio recordings, if you have clean recordings to learn from.

### Example Usage

Suppose you have a big set of audio recordings of African Swallow mating calls. TFASTS can process these calls to learn what African Swallows sound like.

        learn clearly_recorded_audio_files/african_swallows/ african_swallows.rf

Now, you might have a new audio recording that contains an African Swallow amidst noise. You can filter out the noise like so:

        filter -m african_swallows.rf new_noisy_input.wav filtered_output.wav

If all goes well, the output file will contain less noise, and more African Swallow.
The quality of results depends on how many recordings of swallows you have, how much non-Swallow noise exists in the source recordings, and many command-line parameters.


### Other Tools:

spectrogram input.wav output.bmp
	Converts audio files into viewable images.

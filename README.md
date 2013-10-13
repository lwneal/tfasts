## Time-Frequency Audio Segmentation Tool Set

Version 0.1: Extremely buggy, work in progress. Good results are not guaranteed.

TFASTS is a set of command-line tools for manipulating audio in the spectrogram domain.

You can use TFASTS to remove some noise from audio recordings, if you have clean recordings to learn from.

### Example Usage

Suppose you have dozens of clean audio recordings of African Swallow mating calls, like so: 

        clearly_recorded_audio_files/african_swallows/example_1.wav
        clearly_recorded_audio_files/african_swallows/example_2.wav
        ...

TFASTS can process these calls to learn what African Swallows sound like.

        learn -i clearly_recorded_audio_files/african_swallows/ -o african_swallows.rf

Now, suppose you go outside with a microphone and record a flock of swallows as they migrate. 

        noisy/african_swallows/example_1.wav
        noisy/african_swallows/example_2.wav
        ...

Your recordings might contain noise from wind, rain, and passing automobiles. You want to hear less noise, and more African Swallow. TFASTS can remove noise from the recording:

        filter -m african_swallows.rf -i noisy/african_swallows/example_1.wav -o filtered_example_1.wav

If all goes well, the output file will contain less noise, and more African Swallow. The quality of results depends on how many recordings of swallows you have, how much non-Swallow noise exists in the source recordings, and many command-line parameters.


### But I Don't Have Any Clean Recordings!

TFASTS can still help you! Suppose all you have are noisy recordings, containing cars, wind, etc. among the Swallow calls.

        noisy_recordings/african_swallows/example_1.wav
        noisy_recordings/african_swallows/example_2.wav
	...

You can teach TFASTS which sounds in your audio recordings are African Swallows, and which sounds are undesired noise.

First, use the Spectrogram tool to generate viewable BMP-format representations of the audio in each noisy file.

        spectrogram -i noisy_recordings/african_swallows/example_1.wav -o noisy_spectrograms/example_1.bmp
        spectrogram -i noisy_recordings/african_swallows/example_2.wav -o noisy_spectrograms/example_2.bmp
	...

Spectrograms look like this:

![Spectrogram alt text](/demo_specs/PC5_20090703_110000_0040.jpg?raw=true)

Now, open and view each spectrogram. Using an image manipulation program like GIMP, color all the African Swallow calls white, and color all the noise black. Spectrogram labels look like this:

![Spectrogram Label alt text](/demo_labels/PC5_20090703_110000_0040.jpg?raw=true)

Save these "spectrogram labels" in a separate folder:

        spectrogram_labels/example_1.bmp
        spectrogram_labels/example_2.bmp
        ...

Now, run:

        learn -i noisy_recordings/african_swallows/ -l spectrogram_labels/ -o african_swallows.rf

You can now use the 'filter' tool as above to remove noise from recordings.

### What Do I Need To Build It?
Linux requirements:

	sudo apt-get install software-properties-common
        sudo add-apt-repository 'deb http://llvm.org/apt/precise/ llvm-toolchain-precise main'
	wget -O llvm-snapshot.gpg.key http://llvm.org/apt/llvm-snapshot.gpg.key
	sudo apt-key add llvm-snapshot.gpg.key
	sudo apt-get update
	sudo apt-get install clang-3.4 clang-3.4-doc libclang-common-3.4-dev libclang-3.4-dev libclang1-3.4 libclang1-3.4-dbg libllvm-3.4-ocaml-dev libllvm3.4 libllvm3.4-dbg lldb-3.4 llvm-3.4 llvm-3.4-dev llvm-3.4-doc llvm-3.4-examples llvm-3.4-runtime clang-modernize-3.4 clang-format-3.4

OSX requirements: XCode 4.3+ with Command-Line Tools installed.

Windows requirements: MSVC++ 2010 or higher. Configuration is left as an excercise for the reader.

### How Do I Build It?
	git clone http://github.com/lwneal/tfasts
	cd tfasts
        make

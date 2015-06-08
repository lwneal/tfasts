#pragma once

#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <stdint.h>
#include <limits.h>

struct RIFF_WAV_std_head {
	// RIFF Header
	char	ChunkID[4];		// ASCII big-endian 'RIFF'
	int32_t	ChunkSize;		// Filesize minus 8 (in bytes)
	char	Format[4];		// ASCII 'WAVE'

	// WAVE fmt subchunk
	char	Subchunk1ID[4];	// ASCII 'fmt '
	int32_t	Subchunk1Size;	// Size of the rest of the subchunk
	int16_t	AudioFormat;	// PCM=1, IEEEFloat=3, Alaw=6, ulaw=7
	int16_t	NumChannels;	// 1 for mono, 2 for stero, etc
	int32_t	SampleRate;		// 8000 for telephony, 44.1k for CD, etc
	int32_t	ByteRate;		// SampleRate*NumChannels*BitsPerSample/8
	int16_t	BlockAlign;		// Bytes per sample including all channels
	int16_t	BitsPerSample;	// Should be multiple of 8
	// Non-standard headers may include:
	// int16_t ExtraParamSize
	// followed by <ExtraParamSize> bytes of data

	// WAVE data subchunk
	char	Subchunk2ID[4];	// ASCII 'data'
	int32_t	Subchunk2Size;	// NumSamples*NumChannels*BitsPerSample/8
	// Header will be followed by <Subchunk2Size> bytes of data

	bool is_valid_header() {
		if (ChunkID[0] != 'R' || ChunkID[1] != 'I'
			|| ChunkID[2] != 'F' || ChunkID[3] != 'F')
      std::cerr << "Wav header does not start with 'RIFF'" << std::endl;
			return false;
		if (Subchunk1ID[0] != 'f' || Subchunk1ID[1] != 'm'
			|| Subchunk1ID[2] != 't' || Subchunk1ID[3] != ' ')
      std::cerr << "Missing 'fmt' in WAV header" << std::endl;
			return false;
		if (Subchunk2ID[0] != 'd' || Subchunk2ID[1] != 'a'
			|| Subchunk2ID[2] != 't' || Subchunk2ID[3] != 'a')
      std::cerr << "Missing 'data' in WAV header" << std::endl;
			return false;
		return true;
	}
};

// Appends PCM samples as floats between -1 and 1, from the
//  given WAV file, into the given vector container
// Handles RIFF WAV files only, and mixes channels down
//	to mono, additively. 8-bit or 16-bit PCM only
// Assumes vec_type implements push_back(float)
// Returns the number of samples read
template <typename vec_type>
long load_pcm_wav_mono(std::string &filename, vec_type& samples, int &sample_rate) 
{
	long initial_size = samples.size();
	// Open a binary stream for the file
	std::ifstream ifs(filename, std::ios::binary);
	if (!ifs.good())
		std::cerr << "Error opening file " << filename << std::endl;

	// Read and validate the header of the file
	RIFF_WAV_std_head header;
	ifs.read( (char*)&header, sizeof(RIFF_WAV_std_head) );
	if (!ifs.good())
		std::cerr << "Error opening header for file " << filename << std::endl;
	if (!header.is_valid_header())
		std::cerr << "Error parsing header for WAV file " << filename << std::endl;

	// Allocate the required number of samples for the PCM input
	samples.reserve(header.Subchunk2Size / (header.NumChannels * header.BitsPerSample/8) );

	// Read the body of the file into the given vector
	char *buff = new char[header.BlockAlign];
	while ( ifs.read(buff, header.BlockAlign) ) {
		double sample = 0;
		if (header.BitsPerSample == 8)
			for (int8_t *p = (int8_t*)buff; (char*)p < buff + header.NumChannels; p += 1)
				sample += (double)(*p) / CHAR_MAX;
		else if (header.BitsPerSample == 16)
			for (int16_t *p = (int16_t*)buff; (char*)p < buff + header.NumChannels; p += 2)
				sample += (double)(*p) / SHRT_MAX;
		samples.push_back(sample);
	}
	delete buff;

	sample_rate = header.SampleRate;
	return samples.size() - initial_size;
}

#pragma once

#include <vector>
#include <string>
#include <math.h>
#include <iostream>

#include "wav_in/wav_in.h"
#include "wav_in/wav_out.h"
 
using namespace std;

class pWavData
{
public:
	unsigned int sampleRate_;
	unsigned int bitsPerSample_;
	unsigned int channels_;
	vector<float> samples_;
	
	pWavData() {}
	pWavData(int sampleRate, unsigned int bitsPerSample, unsigned int channels) :
		sampleRate_(sampleRate), bitsPerSample_(bitsPerSample), channels_(channels) {}
	
	pWavData(string filename)
	{
		WAV_IN infile(filename.c_str());
		sampleRate_ = int(infile.get_sample_rate_hz());
		bitsPerSample_ = unsigned int(infile.get_bits_per_sample());
		channels_ = infile.get_num_channels();
	
		if( channels_ == 2  ) // if 2 channels, we need to convert to mono
		{
			cout << "CONVERTING STEREO TO MONO!!!" << endl;
			
			vector<float> samplesLR;
			while(infile.more_data_available()) 
				samplesLR.push_back(float(infile.read_current_input() / 32768.0)); // 32768 is the maximum magnitude for a 16bit signed integer
			
			for(unsigned int i = 0; i < samplesLR.size(); i +=2)
			{
				double avg = .5 * (samplesLR[i] + samplesLR[i+1]);
				samples_.push_back(float(avg) );
			}
			
			channels_ = 1;
		}
		else
		{
			// read data from input file into memory
			while(infile.more_data_available()) 
			{
				double sample = infile.read_current_input();
				samples_.push_back(float(sample / 32768.0)); // this is the maximum magnitude for a 16bit signed integer
			}
			
		}
	}
	
	pWavData(string filename, int channel )
	{
		WAV_IN infile(filename.c_str());
		sampleRate_ = (unsigned int)(infile.get_sample_rate_hz());
		bitsPerSample_ = infile.get_bits_per_sample();
		channels_ = infile.get_num_channels();
		
		vector<float> samplesLR;
		while(infile.more_data_available()) 
			samplesLR.push_back(float(infile.read_current_input() / 32768.0)); // 32768 is the maximum magnitude for a 16bit signed integer
		
		
		for(unsigned int i = 0; i < samplesLR.size(); i +=2)
		{
			samples_.push_back(samplesLR[i + channel]);
		}
		
		channels_ = 1;
	}
	
	// split this wav into chunks of duration (seconds)
	void splitIntoChunks(float duration, vector<pWavData>& outChunks)
	{
		cout << "splitting into chunks " << endl;
		int samplesPerChunk = int(sampleRate_ * duration);
		int totalChunks = int(floor((float)samples_.size() / samplesPerChunk));
		
		int currSample = 0;
		for(int i = 0; i < totalChunks; ++i)
		{	
			cout << "chunk #" << i << endl;
			pWavData chunk(sampleRate_, bitsPerSample_, channels_);
			for(int j = 0; j < samplesPerChunk; ++j)
			{
				chunk.samples_.push_back(samples_[currSample]);
				++currSample;
			}
			outChunks.push_back(chunk);
		}
	}
	
	pWavData extractChunk(double start, double duration)
	{
		cout << "splitting into chunks " << endl;
		int samplesPerChunk = int(sampleRate_ * duration);
		
		int currSample = int(sampleRate_ * start);

		pWavData chunk(sampleRate_, bitsPerSample_, channels_);
		for(int j = 0; j < samplesPerChunk; ++j)
		{
			chunk.samples_.push_back(samples_[currSample]);
			++currSample;
		}
		return chunk;

	}
	
	void writeWAV(string filename)
	{
		WAV_OUT outfile(sampleRate_, bitsPerSample_, channels_);
		for(unsigned int i = 0; i < samples_.size(); ++i)
		{
			//cout << "\t" << samples_[i] << endl;
			outfile.write_current_output(samples_[i] * 32768.0);
		}
		outfile.save_wave_file(filename.c_str());
	}
};

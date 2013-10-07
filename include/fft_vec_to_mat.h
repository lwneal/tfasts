#pragma once

#include "FFT.h"
#include <string>
#include <vector>

// spec_out must be a 2D array of reals accessible by [row][col] operators
template<
	typename spec_type,
	typename vector_of_real
>
void real_fft(
	spec_type &spec_out,
	vector_of_real &signal_in,
	int window_size = 512,
	int window_step = 256,
	int hi_pass = 0)
{
	int fft_height = window_size/2 - hi_pass;
	int fft_length = (signal_in.size() - window_size) / window_step;

	double *window_samples = new double[window_size];
	double *fft_power_out = new double[window_size/2];

	spec_out.set_size(fft_height, fft_length);

	for(int frame = 0; frame < fft_length; frame++) {
		// extract the samples in this frame
		for(int i = 0; i < window_size; ++i)
			window_samples[i] = signal_in[i + (window_step*frame)];
		// apply a hamming window to frame samples
		WindowFunc(2, window_size, window_samples); 
		// do an FFT from real samples to magnitudes of complex FFT outputs
		PowerSpectrum(window_size, window_samples, fft_power_out);

		for(int i = 0; i < fft_height; ++i)
			spec_out[i][frame] = fft_power_out[i+hi_pass];
	}
	delete[] window_samples;
	delete[] fft_power_out;
}

template<
	typename spec_type,
	typename vector_of_real
>
void complex_fft(
	spec_type &real_out,
	spec_type &imag_out,
	vector_of_real &signal_in,
	int window_size = 512,
	int window_step = 256,
	int hi_pass = 0)
{
	int fft_height = window_size/2 - hi_pass;
	int fft_length = (signal_in.size() - window_size) / window_step;

	double *window_samples = new double[window_size];
	double *fft_real_out = new double[window_size/2];
	double *fft_imag_out = new double[window_size/2];

	real_out.set_size(fft_height, fft_length);
	imag_out.set_size(fft_height, fft_length);

	for(int frame = 0; frame < fft_length; frame++) {
		// extract the samples in this frame
		for(int i = 0; i < window_size; ++i)
			window_samples[i] = signal_in[i + (window_step*frame)];
		// apply a hamming window to frame samples
		WindowFunc(2, window_size, window_samples); 
		// do an FFT from real samples to complex FFT coefs
		RealFFT(window_size, window_samples, fft_real_out, fft_imag_out);

		for(int i = 0; i < fft_height; ++i) {
			real_out[i][frame] = fft_real_out[i+hi_pass];
			imag_out[i][frame] = fft_imag_out[i+hi_pass];
		}
	}
	delete[] window_samples;
	delete[] fft_real_out;
	delete[] fft_imag_out;
}


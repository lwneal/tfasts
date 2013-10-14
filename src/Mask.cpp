#include "Mask.h"
#include "Grid.h"

#include "fft_vec_to_mat.h"
#include "audio_loader.h"
#include "FFT.h"
#include "pWavData.h"

#include "dlib/image_transforms.h"
#include "dlib/threads.h"
#include "dlib/image_io.h"

using namespace std;

using dlib::rectangle;
using dlib::thread_function;


void Mask::foreach_worker(rectangle rect, pixel_fn_t fn) const {
	for (int yrow = rect.top(); yrow < rect.bottom(); yrow++) {
		for (int xcol = rect.left(); xcol < rect.right(); xcol++) {
			fn(xcol,yrow);
			assert(at(xcol, yrow) == at(xcol, yrow));
		}
	}
}

void Mask::initialize_worker(int start, int end, pixel_initfn_t fn) {
	for (int yrow = start; yrow < end; yrow++) {
		for (int xcol = 0; xcol < width(); xcol++) {
			at(xcol,yrow) = fn(xcol,yrow);
			assert(at(xcol, yrow) == at(xcol, yrow));
		}
	}
}

void Mask::foreach(pixel_fn_t fn, int num_threads) const {
	dlib::thread_pool pool(num_threads);
	for (int i = 0; i < num_threads; i++) {
		int start = (i*height()) / num_threads;
		int end = ((i+1)*height()) / num_threads;
		pool.add_task_by_value([=](){
			foreach_worker(rectangle(0, start, width(), end), fn);
		});
	}
	pool.wait_for_all_tasks();
}

void Mask::foreach(pixel_fn_t fn, 
	dlib::rectangle rect, 
	int num_threads) const 
{
	dlib::thread_pool pool(num_threads);
	for (int i = 0; i < num_threads; i++) {
		int lo = rect.top() + (i*(rect.height()-1)) / num_threads;
		int hi = rect.top() + ((i+1)*(rect.height()-1)) / num_threads;
		pool.add_task_by_value([=](){
			foreach_worker(rectangle(rect.left(), lo, rect.right(), hi), fn);
		});
	}
	pool.wait_for_all_tasks();
}

// The bounds checking slows access considerably
double& Mask::operator()(int xcol, int yrow) {   
	clamp_val(xcol, 0, width()-1);
	clamp_val(yrow, 0, height()-1);
	return (*data)[yrow][xcol];
}
double Mask::operator()(int xcol, int yrow) const {   
	clamp_val(xcol, 0, width()-1);
	clamp_val(yrow, 0, height()-1);
	return (*data)[yrow][xcol];
}

double& Mask::at(int xcol, int yrow) {
	return (*data)[yrow][xcol];
}
double Mask::at(int xcol, int yrow) const {
	return (*data)[yrow][xcol];     
}

// Function to load a mask from a previously-output probability mask
Mask::Mask(std::string &bmp_in):
	data(new dlib::array2d<double>)
{
	dlib::array2d<int> pixel_data;
	dlib::load_bmp(pixel_data, bmp_in);
	data->set_size(pixel_data.nr(), pixel_data.nc());
	foreach([&](int x, int y) {
		at(x,y) = (double)(pixel_data[y][x]) / 255.0;
	});
}

// If you get errors here, check the load_audio file, it might
//	not properly parse the format you're giving it
Mask::Mask(string &wav_fn, int fft_size, int fft_step):
	data(new dlib::array2d<double>)
{
	vector<double> audio;
	load_pcm_wav_mono(wav_fn, audio, sample_rate);
	real_fft(*data, audio, fft_size, fft_step);
}

Mask::Mask(int width, int height, pixel_initfn_t fn, int num_threads):
	data(new dlib::array2d<double>(height,width)) 
{
	dlib::thread_pool pool(num_threads);
	for (int i = 0; i < num_threads; i++) {
		int start = (i*height) / num_threads;
		int end = ((i+1)*height) / num_threads;
		pool.add_task_by_value([=]() {
			initialize_worker(start, end, fn);
		});
	}
	pool.wait_for_all_tasks();
}

// Copy constructor
Mask::Mask(const Mask &copy):
	data(new dlib::array2d<double>(copy.height(),copy.width())) 
{
	copy.foreach([&](int x, int y) {
		at(x,y) = copy(x,y);
	});
}

void Mask::attenuate_wav(string &wav_in, string &wav_out) const {
	if ( wav_out.rfind(".wav") == string::npos 
		&& wav_out.rfind(".WAV") == string::npos)
		wav_out = wav_out.append(".wav");
	
	vector<double> audio;
	int sample_rate;
	load_pcm_wav_mono(wav_in, audio, sample_rate);
	cerr << "WAV Loaded " << audio.size() << " samples at " << sample_rate << "hz" << std::endl;

	double original_volume = 0;
	for (int i = 0; i < audio.size(); i++)
		original_volume += abs(audio[i]);
	original_volume /= audio.size();

	Mask real(width(), height());
	Mask imag(width(), height());
	real.sample_rate = imag.sample_rate = sample_rate;
	int fft_size = 1024;
	int fft_step = 128;
	complex_fft(*(real.data), *(imag.data), audio, fft_size, fft_step);
	
	cerr << "Applying attenuation from mask " << toString() << endl;
	cerr << "\tReal: " << real.toString() << endl;
	cerr << "\tImag " << imag.toString() << endl;

	real.foreach([&](int x, int y) {
		int u = x * width() / real.width();
		int v = y * height() / real.height();
		real.at(x,y) *= at(u,v);
		imag.at(x,y) *= at(u,v);
	});

	cerr << "After attenuation" << endl;
	cerr << "\tReal: " << real.toString() << endl;
	cerr << "\tImag " << imag.toString() << endl;

	// Output at 16-bit mono PCM
	pWavData outWav(sample_rate, 16, 1);

	// Floating point buffers for the FFT library
	double* realIn = new double[fft_size];
	double* imagIn = new double[fft_size];
	double* realOut = new double[fft_size];
	double* imagOut = new double[fft_size];
		
	// For each frame of audio,
	for (int i = 0; i < real.width(); i++) {
		// In order to apply the inverse FFT, we need fft_size coefficients
		//  (so we get a full frame of audio back). To do this, we mirror
		//  the left half of the signal to the right, taking the complex
		//  conjugate of each value. 
		for (int j = 0; j < fft_size/2; j++) {
			realIn[j] = real(i,j);
			imagIn[j] = imag(i,j);
			
			realIn[fft_size - j] = real(i,j);
			imagIn[fft_size - j] = -imag(i,j);
		}
		// Also, the coefficient in the middle (zero frequency) has to be zeroed.
		realIn[fft_size/2 + 1] = 0;
		imagIn[fft_size/2 + 1] = 0;

		// Here, we apply the inverse FFT. Assuming that the conjugate-mirror 
		//  property holds true, the output should consist of fft_size real
		//  coefficients, representing the audio signal in this frame.
		// Imaginary coefficients output should be 0 (+/- epsilon)
		FFT( fft_size, true, realIn, imagIn, realOut, imagOut );
		
		// Here, we fold the time-domain values back into something resembling
		//  the original signal. Each frame overlaps by fft_size - fft_step
		// (Except for the first frame)
		int framesToAdd = i ? fft_step : fft_size;
		outWav.samples_.insert( outWav.samples_.end(), framesToAdd, 0);
		
		for (int m = 0; m < fft_size; m++ ) {
			outWav.samples_[ outWav.samples_.size() - fft_size + m ] += realOut[m];
		}
	}

	double transformed_volume = 0;
	for (int i = 0; i < outWav.samples_.size(); i++) 
		transformed_volume += abs(outWav.samples_[i]);
	transformed_volume /= outWav.samples_.size();

	if (transformed_volume == 0) {
		cerr << "Warning: Empty audio output" << endl;
	} else {
		double adjustment = original_volume / transformed_volume;
		cerr << "Adjusting volume by factor of " << adjustment << endl;
		for (int i = 0; i < outWav.samples_.size(); i++)
			outWav.samples_[i] *= adjustment;
	}

	outWav.writeWAV(wav_out);
	
	delete[] realOut;
	delete[] imagOut;
	delete[] realIn;
	delete[] imagIn;
}

bool Mask::empty() const {
	bool empty = true;
	foreach([&](int x, int y) {
		if (at(x,y) > 0)
			empty = false;
	});
	return empty;
}

// We use a dlib mutex to protect the 'maxval' var
double Mask::get_max() const {
	double maxval = 0;
	dlib::mutex max_mutex;
	foreach([&](int x, int y) {
		if (at(x,y) > maxval) {
			dlib::auto_mutex lock(max_mutex);
			maxval = at(x,y);
		}
	});
	return maxval;
}

double Mask::get_min() const {
	double minval = 0;
	dlib::mutex min_mutex;
	foreach([&](int x, int y) {
		if (at(x,y) < minval) {
			dlib::auto_mutex lock(min_mutex);
			minval = at(x,y);
		}
	});
	return minval;
}

double Mask::get_mean() const {
	double total = 0;
	foreach([&](int x, int y) {
		total += at(x,y);
	}, 1);
	return total / (height()*width());
}

double Mask::get_variance() const {
	double sum = 0;
	double sumsqr = 0;

	foreach([&](int x, int y) {
		double val = at(x,y);
		sum += val;
		sumsqr += val*val;
	}, 1);
	return (sumsqr - sum*sum) / size();
}

Mask Mask::div_by(double div) const {
	return Mask(width(), height(), [&](int x, int y) {
		return at(x,y) / div;
	});
}

Mask Mask::mult_by(double mul) const {
	return Mask(width(), height(), [&](int x, int y) {
		return at(x,y) * mul;
	});
}

Mask Mask::raise_to(double pow) const {
	return Mask(width(), height(), [&](int x, int y) {
		return std::pow(at(x,y), pow);
	});
}

Mask Mask::whitening_filter(double quietest_frames) const {
	vector<double> noise = noise_profile(quietest_frames);
	for (int i = 0; i < noise.size(); i++)
		if (noise[i] == 0) 
			noise[i] = 1;
	return Mask(width(), height(), [&](int x, int y) {
		return at(x,y) / noise[y];
	});
}

Mask Mask::gaussian_blur(double sigma) const {
	dlib::matrix<double, 0, 1> gauss = 
		dlib::create_gaussian_filter<double>(sigma, 201);
	int ksize = gauss.nr();
	Mask buffer(width(), height(), [&](int x, int y) -> double {
		double val = 0;
		for (int i = 0; i < ksize; i++)
			val += gauss(i) * (*this)(x, y - ksize/2 + i);
		return val;
	});
	return Mask(width(), height(), [&](int x, int y) -> double {
		double val = 0;
		for (int i = 0; i < ksize; i++)
			val += gauss(i) * buffer(x - ksize/2 + i, y);
		return val;
	});
}

// Create a Gaussian kernel, but only use the central 
//	[-radius,radius] elements from it
Mask Mask::variance_mask(int radius, double sigma) const {
	dlib::matrix<double, 0, 1> gauss = 
		dlib::create_gaussian_filter<double>(sigma, 201);
	int center = gauss.nr()/2;

	double gauss_total = 0;
	for (int r = -radius; r < radius; r++)
		for (int c = -radius; c < radius; c++)
			gauss_total += gauss(center+r)*gauss(center+c);

	return Mask(width(), height(), [&](int x, int y) -> double {
		double sum = 0, sumsqr = 0;
		for (int ax = x-radius; ax < x+radius; ax++) {
			for (int ay = y-radius; ay < y+radius; ay++) {
				double weightx = gauss(center + (ax - x));
				double weighty = gauss(center + (ay - y));
				double val = (*this)(ax,ay) * weightx * weighty;
				sum += val;
				sumsqr += val*val;
			}
		}
		double mean = sum / gauss_total;
		double variance = (sumsqr - sum*mean) / gauss_total;
		return variance;
	});
}


Mask Mask::normpdf() const {
	vector<double> sums(width(), 0);
	foreach([&](int x, int y) {
		sums[x] += at(x,y);
	});
	return Mask(width(), height(), [&](int x, int y) {
		return at(x,y) / sums[x];
	});
}

Grid Mask::threshold_segment(double thresh) const {
	return Grid(width(), height(), [&](int x, int y) {
		return at(x,y) > thresh ? 1 : 0;
	});
}

// The first of the pair output contains just X coordinate values,
//	while the second contains just Y. Feed them both to the super-
//	pixel segmentation
pair<Mask,Mask> Mask::voronoi_peaks(double sigma) const {
	dlib::matrix<double, 0, 1> gauss = 
		dlib::create_gaussian_filter<double>(sigma, 201);
	int center = gauss.nr()/2;

	auto get_peak = [&](int x, int y) -> pair<int,int> {
		double peak_val = 0;
		pair<int,int> peak;
		for (int i = 0, r = y-center; r < y+center; i++, r++) {
			for (int j = 0, c = x-center; c < x+center; j++, c++) {
				double val = gauss(i)*gauss(j)*(*this)(c,r);
				if (val > peak_val) {
					peak = std::make_pair(c,r);
					peak_val = val;
				}
			}
		}
		return peak;
	};
	return std::make_pair(
		Mask(width(), height(), [&](int x, int y) {
			return (double)get_peak(x,y).first;
		}),
		Mask(width(), height(), [&](int x, int y) {
			return (double)get_peak(x,y).second;
		}));
}

// Returns both gradients (as a pair, X then Y)
std::pair<Mask, Mask> Mask::sobel_gradient() const {
	Mask grad_h(*this);
	Mask grad_v(*this);
	dlib::sobel_edge_detector(*(get_data()), *(grad_h.get_data()), *(grad_v.get_data()) );
	return std::make_pair(grad_h, grad_v);
}

// Returns the horizontal gradient, via a Sobel filter
// A value of -1 corresponds to maximum intensity leftward
// A value of +1 corresponds to maximum intensity right
Mask Mask::sobel_gradient_x() const {	
	dlib::array2d<double> grad_horz(height(), width());
	dlib::array2d<double> grad_vert(height(), width());
	dlib::sobel_edge_detector(*(get_data()), grad_horz, grad_vert);
	return Mask(width(), height(), [&](int x, int y) {
		return grad_horz[y][x];
	});
}


// Returns the vertical gradient, via a Sobel filter
// A value of -1 corresponds to maximum intensity downward
// A value of +1 corresponds to maximum intensity upward
Mask Mask::sobel_gradient_y() const {	
	dlib::array2d<double> grad_horz(height(), width());
	dlib::array2d<double> grad_vert(height(), width());
	dlib::sobel_edge_detector(*(get_data()), grad_horz, grad_vert);
	return Mask(width(), height(), [&](int x, int y) {
		return grad_vert[y][x];
	});
}

// Computes the average spectral energy at each frequency in the
//  softest <percent_lowest> frames. Returns a 1D vector where
//	each element represents the relative amount of noise energy
//	in that frequency range. (assuming the softest frames consist
//	of only noise.
// Note 1/25/11: Using root-mean-square now instead of mean
vector<double> Mask::noise_profile(double percent_lowest) const
{	
	vector<double> profile_out;
	int height = this->height();
	int width = this->width();

	// First, create a vector storing the index and total amplitude of each frame
	std::vector< pair<double, int> > column_amplitude(width);
	for (int i = 0; i < width; i++) {
		double col_total = .0;
		for (int j = 0; j < height; j++)
			col_total += at(i,j);
		column_amplitude[i] = pair<double, int>(col_total, i);
	}

	// Sort with the STL default comparison (a.first < b.first)
	sort(column_amplitude.begin(), column_amplitude.end() );

	// Initialize the input profile to a zero vector
	profile_out.resize(height);
	for (int i = 0; i < height; i++)
		profile_out[i] = 0;

	// Now, compute the average of the softest N percent of frames
	int num_low_frames = width * percent_lowest;
	for (int i = 0; i < num_low_frames; i++) {
		int index = column_amplitude[i].second;
		for (int j = 0; j < height; j++) {
			profile_out[j] += at(index,j)*at(index,j);
		}
	}

	// Take sqrt and normalize for the number of frames used
	for (int i = 0; i < height; i++)
		profile_out[i] = sqrt(profile_out[i]) / num_low_frames;
	return profile_out;
}

void Mask::band_pass(int hi_pass_hz, int low_pass_hz) {
	if (sample_rate <= 0) {
		std::cerr << "Warning: No sample rate given for band pass: assuming 16khz" << std::endl;
		sample_rate = 16000;
	}
		
	if (hi_pass_hz > 0) {
		int band_pass_px = (hi_pass_hz * height()) / sample_rate;
		foreach([&](int x, int y) {
			if (y < band_pass_px) at(x,y) = 0;
		});
	}
	if (low_pass_hz > 0) {
		int band_pass_px = (low_pass_hz * height()) / sample_rate;
		foreach([&](int x, int y) {
			if (y > band_pass_px) at(x,y) = 0;
		});
	}
}

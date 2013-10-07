#include "Mask.h"
#include "Image.h"
#include "Utility.h"
#include "pRandomForest.h"

#include "dlib/cmd_line_parser.h"
#include "dlib/image_io.h"

#include <string>
#include <iostream>

using namespace std;

typedef dlib::array2d<dlib::rgb_pixel> ImageRGB;
typedef dlib::rgb_pixel Pixel;

struct Options {
	int fft_width = 512;
	int fft_step = 256;
	int hi_pass_hz = 1000;
	string input_path;
	string output_path;
	vector<string> input_files() {
		vector<string> files = get_full_filenames_from_dir(input_path, "");
		if (files.size() == 0)
			files.push_back(input_path);
		return files;
	}
	bool good() {
		return input_path.size() && output_path.size() 
			&& fft_width > 0 && fft_step > 0
			&& hi_pass_hz >= 0;
	}
};

Options spectrogram_parse_args(int argc, char *argv[]) {
	dlib::cmd_line_parser<char>::check_1a_c parser;
	parser.add_option("h", "Display this help message");
	parser.add_option("i", "An input .wav audio file", 1);
	parser.add_option("o", "Filename for output .bmp spectrogram",1);
	parser.add_option("w", "Integer FFT width (2x the output spectrogram image height). Default 512", 1);
	parser.add_option("s", "FFT Step (smaller step sizes result in a wider image). Default 256", 1);
	parser.add_option("p", "High-Pass cutoff in Hz. Default 1000", 1);
	parser.parse(argc, argv);

	Options opts;
	if (parser.number_of_arguments() > 0)
		opts.input_path = parser[0];
	if (parser.number_of_arguments() > 1)
		opts.output_path = parser[1];

	if (parser.option("i") && parser.option("i").count() > 0)
		opts.input_path = parser.option("i").argument();
	if (parser.option("o") && parser.option("o").count() > 0)
		opts.output_path = parser.option("o").argument();
	if (parser.option("w") && parser.option("w").count() > 0)
		opts.fft_width = dlib::sa = parser.option("w").argument();
	if (parser.option("s") && parser.option("s").count() > 0)
		opts.fft_step = dlib::sa = parser.option("s").argument();
	if (parser.option("p") && parser.option("p").count() > 0)
		opts.hi_pass_hz = dlib::sa = parser.option("p").argument();
		
	if (!opts.good()) {
		cout << "\tspectrogram sound.wav output.bmp" << endl;
		parser.print_options();
		exit(1);
	}
	return opts;
}

int main(int argc, char *argv[]) {
	Options opt = spectrogram_parse_args(argc, argv);

	Mask spec(opt.input_path, opt.fft_width, opt.fft_step);

	int band_pass_px = (opt.hi_pass_hz * spec.height()) / 16000;
	spec = Mask(spec.width(), spec.height(), [&](int x, int y) {
		return y < band_pass_px ? 0 : spec(x,y);
	});

	spec = spec.whitening_filter();

	spec = spec.norm_to_max();

	spec = Mask(spec.width(), spec.height(), [&](int x, int y) {
		return sqrt(spec(x,y));
	});

	Image img(spec);
	cout << opt.input_path << " " << img.width() << "," << img.height() << endl;
	img.save(opt.output_path);
	return 0;
}

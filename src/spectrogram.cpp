#include "Mask.h"
#include "Image.h"
#include "pRandomForest.h"

#include "dlib/cmd_line_parser.h"
#include "dlib/image_io.h"

#include <string>
#include <iostream>

using namespace std;

typedef dlib::cmd_line_parser<char>::check_1a_c ArgParser;
typedef dlib::array2d<dlib::rgb_pixel> ImageRGB;
typedef dlib::rgb_pixel Pixel;

struct options {
	int fft_width = 512;
	int fft_step = 256;
	int high_pass_hz = 1000;
};

void spectrogram_parse_args(ArgParser &parser, int argc, char *argv[]) {
	parser.add_option("h", "Display this help message");
	parser.add_option("i", "An input .wav audio file", 1);
	parser.add_option("o", "Filename for output .bmp spectrogram",1);
	parser.parse(argc, argv);

	if (parser.option("h") || !parser.option("o")) {
		cout << "spectrogram:" << endl;
		cout << "\tspectrogram -i input_audio.wav -o output_img.bmp" << endl;
		parser.print_options();
		exit(1);
	}
}

void save_img(Mask &spec, string filename) {
	cout << "Saving image size " << spec.width() << "," << spec.height() << " to " << filename << endl;

	ImageRGB save_img(spec.height(), spec.width());
	spec.foreach([&](int x, int y) {
		double val = 255.0 * spec.at(x,y);
		save_img[y][x] = Pixel(val,val,val);
	});
	dlib::save_bmp(save_img, filename);
}

int main(int argc, char *argv[]) {
	ArgParser args;
	spectrogram_parse_args(args, argc, argv);

	string fn_in = args.option("i").argument();
	string fn_out = args.option("o").argument();

	int fft_width = 512;
	int fft_step = 256;
	int high_pass_hz = 1000;

	Mask spec(fn_in, fft_width, fft_step);

	int band_pass_px = (high_pass_hz * spec.height()) / 16000;
	spec = Mask(spec.width(), spec.height(), [&](int x, int y) {
		return y < band_pass_px ? 0 : spec(x,y);
	});

	spec = spec.whitening_filter();

	spec = spec.norm_to_max();

	spec = Mask(spec.width(), spec.height(), [&](int x, int y) {
		return sqrt(spec(x,y));
	});

	Image img(spec);
	cout << fn_out << " " << img.width() << "," << img.height() << endl;
	img.save(fn_out);
	return 0;
}

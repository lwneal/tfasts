#include "pRandomForest.h"
#include "Mask.h"

#include "dlib/cmd_line_parser.h"
#include "dlib/image_io.h"

#include <string>
#include <iostream>

using namespace std;

typedef dlib::cmd_line_parser<char>::check_1a_c ArgParser;
typedef dlib::array2d<dlib::rgb_pixel> ImageRGB;
typedef dlib::rgb_pixel Pixel;

void spectrogram_parse_args(ArgParser &parser, int argc, char *argv[]) {
	parser.add_option("h", "Display this help message");
	parser.add_option("i", "An input .wav audio file", 1);
	parser.add_option("o", "Filename for output .bmp spectrogram",1);
	parser.parse(argc, argv);

	if (parser.option("h") || !parser.option("o")) {
		cout << "TFASTS spectrogram usage:" << endl;
		cout << "\tspectrogram -i input_audio.wav -o output_img.bmp" << endl;
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
	int band_pass_px = 12;

	Mask spec(fn_in, 512, 256);

	spec = Mask(spec.width(), spec.height(), [&](int x, int y) {
		return y < band_pass_px ? 0 : spec(x,y);
	});

	spec = spec.whitening_filter();

	spec = spec.div_by(spec.get_max());

	spec = Mask(spec.width(), spec.height(), [&](int x, int y) {
		return sqrt(spec(x,y));
	});

	save_img(spec, fn_out);
	return 0;
}

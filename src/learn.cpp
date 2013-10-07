#include "Mask.h"
#include "Image.h"
#include "Utility.h"
#include "pRandomForest.h"

#include "dlib/cmd_line_parser.h"
#include "dlib/image_io.h"

#include <vector>
#include <string>
#include <iostream>

using namespace std;

typedef dlib::array2d<dlib::rgb_pixel> ImageRGB;
typedef dlib::rgb_pixel Pixel;

struct Options {
	int fft_width = 512;
	int fft_step = 256;
	int hi_pass_hz = 300;
	string input_path;
	string output_path;
	vector<string> get_filenames() {
		return get_full_filenames_from_dir(input_path, "wav");
	}
	bool good() {
		return input_path.size() && output_path.size() 
			&& fft_width > 0 && fft_step > 0
			&& hi_pass_hz >= 0 && get_filenames().size() > 0;
	}
};

Options learn_parse_args(int argc, char *argv[]) {
	dlib::cmd_line_parser<char>::check_1a_c parser;
	parser.add_option("h", "Display this help message");
	parser.add_option("i", "An input .wav audio file", 1);
	parser.add_option("o", "Filename for output .rf model file",1);
	parser.add_option("w", "Integer FFT width (2x the output spectrogram image height). Default 512", 1);
	parser.add_option("s", "FFT Step (smaller step sizes result in a wider image). Default 256", 1);
	parser.add_option("p", "High-Pass cutoff in Hz. Default 300", 1);
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
		cout << "\tlearn input_dir/ output.rf" << endl;
		parser.print_options();
		exit(1);
	}
	return opts;
}

pExample get_example(Options &opt, Mask &mask, Grid &labels, int x, int y) {
	vector<float> features;
	features.push_back(y);
	for (int u = x - 4; u <= x + 4; u++)
		for (int v = y - 4; v <= y + 4; v++)
			features.push_back(mask(u,v));
	return pExample(features, labels(x, y));
}

int main(int argc, char *argv[]) {
	Options opt = learn_parse_args(argc, argv);

	vector<string> files_in = opt.get_filenames();

	int num_trees = 10;
	int num_classes = 2;
	int examples_per_file = 1000;

	vector<pExample> training;

	for (int i = 0; i < files_in.size(); i++) {
		cout << "Scanning " << i << "/" << files_in.size() << " " << files_in[i] << endl;
		Mask spec(files_in[i], opt.fft_width, opt.fft_step);

		// TODO: Load a binary mask, or bootstrap one with dumb segmentation
		Grid segmentation(spec.width(), spec.height(), [&](int x, int y) {
			return rand() % 2;
		});

		for (int j = 0; j < examples_per_file; j++) {
			int x = rand() % spec.width();
			int y = rand() % spec.height();

			cout << "Getting example at coord " << x << "," << y << endl;
			training.push_back(get_example(opt, spec, segmentation, x, y));
		}
	}

	int feature_dim = training[0].featureVector_.size();
	pRandomForest rf(num_trees, num_classes, feature_dim, training);
	rf.save(opt.output_path);
	return 0;
}

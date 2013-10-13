#include "Mask.h"
#include "Image.h"
#include "Utility.h"
#include "Features.h"
#include "pRandomForest.h"

#include "dlib/cmd_line_parser.h"
#include "dlib/image_io.h"

#include <vector>
#include <string>
#include <iostream>

using namespace std;

struct Options {
	int fft_width = 512;
	int fft_step = 256;
	int hi_pass_hz = 300;
	int k_examples = 10;
	string input_path;
	string output_path;
	string label_path;
	vector<string> get_filenames() {
		return get_full_filenames_from_dir(input_path, "wav");
	}
	vector<string> get_label_filenames() {
		return get_full_filenames_from_dir(label_path, "bmp");
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
	parser.add_option("i", "An input directory containing .wav audio", 1);
	parser.add_option("o", "Filename for output .rf model file",1);
	parser.add_option("l", "An input directory containing .bmp spectrogram labels, with filenames that match each .wav audio file.",1);
	parser.add_option("w", "Integer FFT width (2x the output spectrogram image height). Default 512", 1);
	parser.add_option("s", "FFT Step (smaller step sizes result in a wider image). Default 256", 1);
	parser.add_option("p", "High-Pass cutoff in Hz. Default 300", 1);
	parser.add_option("e", "Number of examples to train from (in thousands). More examples increases accuracy. Default 10", 1);
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
	if (parser.option("l") && parser.option("l").count() > 0)
		opts.label_path = parser.option("l").argument();
	if (parser.option("w") && parser.option("w").count() > 0)
		opts.fft_width = dlib::sa = parser.option("w").argument();
	if (parser.option("s") && parser.option("s").count() > 0)
		opts.fft_step = dlib::sa = parser.option("s").argument();
	if (parser.option("p") && parser.option("p").count() > 0)
		opts.hi_pass_hz = dlib::sa = parser.option("p").argument();
	if (parser.option("e") && parser.option("e").count() > 0)
		opts.k_examples = dlib::sa = parser.option("e").argument();
		
	if (!opts.good()) {
		cout << "\tlearn input_dir/ output.rf" << endl;
		parser.print_options();
		exit(1);
	}
	return opts;
}

// Automatically create a spectrogram label, assuming all loud regions 
// in the input are positive examples
Mask bootstrap_mask(Mask &spec) {
	Mask bootstrap = spec.gaussian_blur(10.0);
	bootstrap = bootstrap.norm_to_mean(1.0);
	bootstrap.foreach([&](int x, int y) {
		bootstrap.at(x,y) = bootstrap.at(x,y) > 1.5;
	});
	if (!bootstrap.empty())
		bootstrap = bootstrap.norm_to_max();
	return bootstrap;
}

int main(int argc, char *argv[]) {
	Options opt = learn_parse_args(argc, argv);

	int num_trees = 10;

	vector<string> files_in = opt.get_filenames();
	vector<string> labels_in;
	
	if (opt.label_path.size()) {
		cout << "Loading spectrogram labels from path " << opt.label_path << endl;
		labels_in = opt.get_label_filenames();
		cout << "Matching " << labels_in.size() << " labels to " << files_in.size() << " input audio files" << endl;
		if (labels_in.size() != files_in.size()) {
			cerr << "ERROR: Number of labels does not match audio input" << endl;
			exit(1);
		}
	}


	vector<pExample> training;
	int i = 0, N = opt.k_examples * 1000;
	int examples_per_file = double(N) / files_in.size();
	while (true) {
		cout << "Scanning file " << i+1 << "/" << files_in.size() << "...";

		Mask spec(files_in[i], opt.fft_width, opt.fft_step);
		spec = preprocess_spec_icassp(spec, opt.hi_pass_hz);

		// If the user supplied labels, use them. Else, generate bootstrap labels.
		Mask labels = labels_in.size() ? Mask(labels_in[i]) : bootstrap_mask(spec);
		
		for (int j = 0; j < examples_per_file; j++) {
			int x = rand() % spec.width();
			int y = rand() % spec.height();

			vector<float> feature(extract_feature_perpixel_icassp(spec, x, y));
			int label = labels.at(x,y) > 0;
			pExample example(feature, label);
			training.push_back(example);

			if (training.size() >= N) break;
		}

		double pct = 100 * double(training.size()) / N;
		cout << "\tprogress " << pct << "\%" << endl;
		i = (i + 1) % files_in.size();
		if (training.size() >= N) break;
	}

	long num_positive = 0;
	for (int i = 0; i < training.size(); i++)
		num_positive += (training[i].classLabel_ > 0);
	cout << "Learned " << num_positive << " positive / " << (training.size() - num_positive) << " examples " << endl;

	int num_classes = 2;
	int feature_dim = training[0].featureVector_.size();
	cout << "Building classifier..." << endl;
	pRandomForest rf(num_trees, num_classes, feature_dim, training);
	cout << endl;
	rf.save(opt.output_path);
	return 0;
}

#include "pRandomForest.h"
#include "Mask.h"

#include "dlib/cmd_line_parser.h"

#include <string>
#include <iostream>

using namespace std;

typedef dlib::cmd_line_parser<char>::check_1a_c ArgParser;


pRandomForest makeTree() {
  int num_trees = 10;
  int num_classes = 2;
  int feature_dim = 4;

  vector<pExample> training;
  pRandomForest *rf = new pRandomForest(num_trees, num_classes, feature_dim, training);
  return *rf;
}

void spectrogram_parse_args(ArgParser &parser, int argc, char *argv[]) {
	parser.add_option("h", "Display this help message");
	parser.add_option("i", "An input .wav audio file", 1);
	parser.add_option("o", "Filename for output .bmp spectrogram");
	parser.parse(argc, argv);

	if (parser.option("h") || !parser.option("o")) {
		cout << "TFASTS spectrogram usage:" << endl;
		cout << "\tspectrogram -i input_audio.wav -o output_img.bmp" << endl;
		exit(1);
	}
}

int main(int argc, char *argv[]) {
	ArgParser args;
	spectrogram_parse_args(args, argc, argv);

	string fn_in = args.option("i").argument();
	string fn_out = args.option("o").argument();

	Mask spec(fn_in);

	cout << "Saving image size " << spec.width() << "," << spec.height() << " to " << fn_out << endl;

	return 0;
}

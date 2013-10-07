/****** NOTICE: LIMITATIONS ON USE AND DISTRIBUTION ********\
 
  This software is provided on an as-is basis, it may be 
  distributed in an unlimited fashion without charge provided
  that it is used for scholarly purposes - it is not for 
  commercial use or profit. This notice must remain unaltered. 
 
  Software by Dr Fred DePiero - CalPoly State University
 
\******************** END OF NOTICE ************************/ 

#ifndef INCLUDE_WAV_IN
#define INCLUDE_WAV_IN
 
 
class WAV_IN
{
 
public:
 
   WAV_IN(const char *wav_file_name);    
   ~WAV_IN();
   
   // routine for reading one sample from a (previously loaded) wave file 
   //  returns current sample as a double 
   double read_current_input();
 
   // determines end-of-file condition, returns 1==true if more data ready 
   int more_data_available();
 
   // returns number of samples in file 
   long int get_num_samples();
 
   // reports number of channels (1==mono, 2==stereo) 
   int get_num_channels();
 
   // reports the number of bits in each sample 
   int get_bits_per_sample();
 
   // reports sample rate in Hz 
   double get_sample_rate_hz();
 
   double fs_hz;
   int bits_per_sample;
   int num_ch;
 
   double *g_wdata_in;
   int g_num_isamp;
   long int g_max_isamp;

};
 
#endif 
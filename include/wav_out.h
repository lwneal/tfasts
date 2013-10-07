/****** NOTICE: LIMITATIONS ON USE AND DISTRIBUTION ********\
 
  This software is provided on an as-is basis, it may be 
  distributed in an unlimited fashion without charge provided
  that it is used for scholarly purposes - it is not for 
  commercial use or profit. This notice must remain unaltered. 
 
  Software by Dr Fred DePiero - CalPoly State University
 
\******************** END OF NOTICE ************************/ 

#ifndef INCLUDE_WAV_OUT
#define INCLUDE_WAV_OUT

class WAV_IN;
 
class WAV_OUT
{
 
public:
 
   // create a new wav_out with given parameters 
   //  note: soundcards typically support a limited range of values! 
   //        hence the next constructor is safer: WAV_OUT(WAV_IN *wav); 
   WAV_OUT(double fs_hz,int bits_per_sample,int num_ch);
   
   // create a wav_out with the same parameters as a given wav_in 
   WAV_OUT(WAV_IN *wav_in);   
 
   ~WAV_OUT();
   
   // routine for writing one output sample 
   //  samples are stored in a buffer, until save_wave_file() is called 
   //  returns 0 on success 
   int write_current_output(double ooo);
 
   // routine for saving a wave file. 
   //  returns 0 on success, negative value on error 
   int save_wave_file(const char *wav_file_name);
 
      
   double fs_hz;
   int bits_per_sample;
   int num_ch;
 
   double *g_wdata_out;
   int g_num_osamp;
   long int g_max_osamp;
 
};
 
#endif 
/****** NOTICE: LIMITATIONS ON USE AND DISTRIBUTION ********\
 
Ê This software is provided on an as-is basis, it may be 
Ê distributed in an unlimited fashion without charge provided
Ê that it is used for scholarly purposes - it is not for 
Ê commercial use or profit. This notice must remain unaltered. 
 
Ê Software by Dr Fred DePiero - CalPoly State University
 
\******************** END OF NOTICE ************************/ 

// header of wav file 
typedef struct{
   char rID[4];            // 'RIFF' 
   int rLen;
      
   char wID[4];            // 'WAVE' 
      
   char fId[4];            // 'fmt ' 
   int pcm_header_len;   // varies... 
   short wFormatTag;
   short nChannels;      // 1,2 for stereo data is (l,r) pairs 
   int nSamplesPerSec;
   int nAvgBytesPerSec;
   short nBlockAlign;      
   short nBitsPerSample;
}   WAV_HDR;
 
   
// header of wav file 
typedef struct{
   char dId[4];            // 'data' or 'fact' 
   int dLen;
//   unsigned char *data; 
}   CHUNK_HDR;
 

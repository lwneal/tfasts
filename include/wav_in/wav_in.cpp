/****** NOTICE: LIMITATIONS ON USE AND DISTRIBUTION ********\
 
  This software is provided on an as-is basis, it may be 
  distributed in an unlimited fashion without charge provided
  that it is used for scholarly purposes - it is not for 
  commercial use or profit. This notice must remain unaltered. 
 
  Software by Dr Fred DePiero - CalPoly State University
 
\******************** END OF NOTICE ************************/ 

#include <cstdio>
#include <cstdlib>
#include <cmath>
 
#include "f_err.h"
#include "f_ptch.h"
 
#include "wav_def.h"
#include "wav_in.h"
 
 
 
 
 
 
/********************************************\
\********************************************/ 
long int WAV_IN::get_num_samples(){ return g_max_isamp; }
 
/********************************************\
\********************************************/ 
int WAV_IN::get_num_channels(){ return num_ch; }
 
/********************************************\
\********************************************/ 
int WAV_IN::get_bits_per_sample(){ return bits_per_sample; }
 
/********************************************\
\********************************************/ 
double WAV_IN::get_sample_rate_hz(){ return fs_hz; }
 
/********************************************\
\********************************************/ 
int WAV_IN::more_data_available()
{ 
   if(g_num_isamp>=g_max_isamp) return 0;
 
   return 1;
}
 
 
 
/**********************************************************
**********************************************************/ 
double WAV_IN::read_current_input()
{ 
   if( (g_wdata_in==NULL) || (g_max_isamp<=0) || (g_num_isamp<0) )
   {
      printf("input file not ready (or not loaded)!!!\n");
      exit(1);
   }
   if(g_num_isamp>=g_max_isamp)
   {
      printf("attempt to read past end of input buffer!\n");
      exit(1);
   }
 
   return( g_wdata_in[g_num_isamp++] );
}
 
 
 
 
 
/********************************************\
\********************************************/ 
WAV_IN::WAV_IN(const char *file_name)
{   
   int i;
   FILE *fw;
   unsigned int wstat;
   char obuff[80];
 
   WAV_HDR *wav;
   CHUNK_HDR *chk;
   short int *uptr;
   unsigned char *cptr;
   int sflag;
   long int rmore;
 
   char *wbuff;
   int wbuff_len;
 
   // set defaults 
   g_wdata_in = NULL;
   g_num_isamp = 0;
   g_max_isamp = 0;
 
   // allocate wav header 
   wav = new WAV_HDR;
   chk = new CHUNK_HDR;
   if(wav==NULL){ printf("cant new headers\n"); exit(-1); }
   if(chk==NULL){ printf("cant new headers\n"); exit(-1); }
 
   /* open wav file */ 
   fw = fopen(file_name,"rb");
   if(fw==NULL){ printf("cant open wav file\n"); exit(-1); }
 
   /* read riff/wav header */ 
   wstat = fread((void *)wav,sizeof(WAV_HDR),(size_t)1,fw);
   if(wstat!=1){ printf("cant read wav\n"); exit(-1); }
 
   // check format of header 
   for(i=0;i<4;i++) obuff[i] = wav->rID[i];
   obuff[4] = 0;
   if(strcmp(obuff,"RIFF")!=0){ printf("bad RIFF format\n"); exit(-1); }
 
   for(i=0;i<4;i++) obuff[i] = wav->wID[i];
   obuff[4] = 0;
   if(strcmp(obuff,"WAVE")!=0){ 
	printf("bad WAVE format: %c %c %c %c\n", obuff[0], obuff[1], obuff[2], obuff[3]); 
	exit(-1); }
 
   for(i=0;i<3;i++) obuff[i] = wav->fId[i];
   obuff[3] = 0;
   if(strcmp(obuff,"fmt")!=0){ printf("bad fmt format\n"); exit(-1); }
 
   if(wav->wFormatTag!=1){ printf("bad wav wFormatTag\n"); exit(-1); }
   
   if( (wav->nBitsPerSample != 16) && (wav->nBitsPerSample != 8) ){
      printf("bad wav nBitsPerSample\n"); exit(-1); }
 
 
   // skip over any remaining portion of wav header 
   rmore = wav->pcm_header_len - (sizeof(WAV_HDR) - 20);
   wstat = fseek(fw,rmore,SEEK_CUR);
   if(wstat!=0){ printf("cant seek\n"); exit(-1); }
 
 
   // read chunks until a 'data' chunk is found 
   sflag = 1;
   while(sflag!=0){
 
      // check attempts 
      if(sflag>10){ printf("too many chunks\n"); exit(-1); }
 
      // read chunk header 
      wstat = fread((void *)chk,sizeof(CHUNK_HDR),(size_t)1,fw);
      if(wstat!=1){ printf("cant read chunk\n"); exit(-1); }
 
      // check chunk type 
      for(i=0;i<4;i++) obuff[i] = chk->dId[i];
      obuff[4] = 0;
      if(strcmp(obuff,"data")==0) break;
      
      // skip over chunk 
      sflag++; 
      wstat = fseek(fw,chk->dLen,SEEK_CUR);
      if(wstat!=0){ printf("cant seek\n"); exit(-1); }
   }
 
   /* find length of remaining data */ 
   wbuff_len = chk->dLen;
 
   // find number of samples 
   g_max_isamp = chk->dLen;
   g_max_isamp /= wav->nBitsPerSample / 8;
 
   /* allocate new buffers */ 
   wbuff = new char [wbuff_len];
   if(wbuff==NULL){ printf("cant alloc\n"); exit(-1); }
 
//   if(g_wdata_in!=NULL) delete g_wdata_in; 
   g_wdata_in = new double [g_max_isamp];
   if(g_wdata_in==NULL){ printf("cant alloc\n"); exit(-1); }
 
 
   /* read signal data */ 
   wstat = fread((void *)wbuff,wbuff_len,(size_t)1,fw);
   if(wstat!=1){ printf("cant read wbuff\n"); exit(-1); }
 
   // convert data 
   if(wav->nBitsPerSample == 16){
      uptr = (short *) wbuff;
      for(i=0;i<g_max_isamp;i++) g_wdata_in[i] = (double) (uptr[i]);
   }
   else{
      cptr = (unsigned char *) wbuff;
      for(i=0;i<g_max_isamp;i++) g_wdata_in[i] = (double) (cptr[i]);
   }
 
   // save demographics 
   fs_hz = (double) (wav->nSamplesPerSec);
   bits_per_sample = wav->nBitsPerSample;
   num_ch = wav->nChannels;
   /*
   printf("\nLoaded WAV File: %s\n",file_name);
   printf(" Sample Rate = %1.0lf (Hz)\n",fs_hz);
   printf(" Number of Samples = %ld\n",g_max_isamp);
   printf(" Bits Per Sample = %d\n",bits_per_sample);
   printf(" Number of Channels = %d\n\n",num_ch);
 */
   // reset buffer stream index 
   g_num_isamp = 0;
 
   // be polite - clean up 
   if(wbuff!=NULL) delete wbuff;
   if(wav!=NULL) delete wav;
   if(chk!=NULL) delete chk;
   fclose(fw);
 
   return;
 
/* WAV_IN::WAV_IN() */}
 
 
 
 
 
 
/********************************************\
\********************************************/ 
WAV_IN::~WAV_IN(){ delete[] g_wdata_in; }
 
 

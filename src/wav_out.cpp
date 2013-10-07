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
#include "wav_out.h"
#include "wav_in.h"
 
 
 
 
 
/**********************************************************
**********************************************************/ 
int WAV_OUT::write_current_output(double ooo)
{
   int i;
   double *tmp = NULL;
 
   // alloc initial buffer 
   if(g_wdata_out==NULL)
   {
      g_max_osamp = 1024;
 
      g_wdata_out = new double [g_max_osamp];
      for(i=0;i<g_num_osamp;i++) g_wdata_out[i] = 0.0;
 
      if(g_wdata_out==NULL){ printf("cant alloc in WAV_OUT\n"); exit(-1); }
   }
 
   // enlarge buffer 
   if(g_num_osamp>=g_max_osamp)
   {
      g_max_osamp *= 2;
      tmp = new double [g_max_osamp];
      if(tmp==NULL){  printf("cant realloc in WAV_OUT\n");  exit(-1); }
 
      // copy over 
      for(i=0;i<g_num_osamp;i++) tmp[i] = g_wdata_out[i];
      for(i=g_num_osamp;i<g_max_osamp;i++) tmp[i] = 0.0;
 
      // swap buffers 
      delete g_wdata_out;
      g_wdata_out = tmp;
   }
 
   // buffer input data 
   g_wdata_out[g_num_osamp++] = ooo;
 
   // be polite 
   return 0;
 
/* int WAV_OUT::write(double ooo) */}
 
 
 
 
 
 
/**********************************************************
**********************************************************/ 
int WAV_OUT::save_wave_file(const char *fname)
{
   FILE *fw;
   unsigned int wstat;
   int i;
   char obuff[80];
 
   WAV_HDR *wav;
   CHUNK_HDR *chk;
   char *wbuff;
   int wbuff_len;
 
   short int *uptr;
   double ttt;
   double max_uuu =  (65536.0 / 2.0) - 1.0;
   double min_uuu = -(65536.0 / 2.0);
 
   unsigned char *cptr;
   double max_ccc = 256.0;
   double min_ccc = 0.0;
 
   if(g_num_osamp<=0) printf("warning, no new data written to output\n");
 
   // allocate wav header 
   wav = new WAV_HDR;
   chk = new CHUNK_HDR;
   if(wav==NULL){ printf("cant new headers\n"); exit(-1); }
   if(chk==NULL){ printf("cant new headers\n"); exit(-1); }
 
   /* allocate new data buffers */ 
   wbuff_len = g_num_osamp * bits_per_sample / 8;
   wbuff = new char [wbuff_len];
   if(wbuff==NULL){ printf("cant alloc\n"); exit(-1); }
 
   // setup wav header 
   sprintf(obuff,"RIFF");
   for(i=0;i<4;i++) wav->rID[i] = obuff[i];
 
   sprintf(obuff,"WAVE");
   for(i=0;i<4;i++) wav->wID[i] = obuff[i];
 
   sprintf(obuff,"fmt ");
   for(i=0;i<4;i++) wav->fId[i] = obuff[i];
 
   wav->nBitsPerSample = bits_per_sample;
   wav->nSamplesPerSec = (int) fs_hz;
   wav->nAvgBytesPerSec = (int) fs_hz;
   wav->nAvgBytesPerSec *= bits_per_sample / 8;
   wav->nAvgBytesPerSec *= num_ch;
   wav->nChannels = num_ch;
   
   wav->pcm_header_len = 16;
   wav->wFormatTag = 1;
   wav->rLen = sizeof(WAV_HDR) + sizeof(CHUNK_HDR) + wbuff_len;
   wav->nBlockAlign = num_ch * bits_per_sample / 8;
 
 
   // setup chunk header 
   sprintf(obuff,"data");
   for(i=0;i<4;i++) chk->dId[i] = obuff[i];
 
   chk->dLen = wbuff_len;
 
 
   // convert data 
   if(bits_per_sample == 16){
      uptr = (short *) wbuff;
      for(i=0;i<g_num_osamp;i++){
         ttt = g_wdata_out[i];
         if(ttt>max_uuu) ttt = max_uuu;
         if(ttt<min_uuu) ttt = min_uuu;
         uptr[i] = (short int) ttt;
      }
   }
   else if(bits_per_sample == 8){
      cptr = (unsigned char *) wbuff;
      for(i=0;i<g_num_osamp;i++){
         ttt = g_wdata_out[i];
         if(ttt>max_ccc) ttt = max_ccc;
         if(ttt<min_ccc) ttt = min_ccc;
         cptr[i] = (unsigned char) ttt;
      }
   }
   else{ printf("bunk bits_per_sample\n"); exit(-1); }
 
 
   /* open wav file */ 
   fw = fopen(fname,"wb");
   if(fw==NULL){ printf("cant open wav file\n"); exit(-1); }
 
 
   /* write riff/wav header */ 
   wstat = fwrite((void *)wav,sizeof(WAV_HDR),(size_t)1,fw);
   if(wstat!=1){ printf("cant write wav\n"); exit(-1); }
 
   /* write chunk header */ 
   wstat = fwrite((void *)chk,sizeof(CHUNK_HDR),(size_t)1,fw);
   if(wstat!=1){ printf("cant write chk\n"); exit(-1); }
 
   /* write data */ 
   wstat = fwrite((void *)wbuff,wbuff_len,(size_t)1,fw);
   if(wstat!=1){ printf("cant write wbuff\n"); exit(-1); }
 
 
   printf("\nSaved WAV File: %s\n",fname);
   printf(" Sample Rate = %1.0lf (Hz)\n",fs_hz);
   printf(" Number of Samples = %d\n",g_num_osamp);
   printf(" Bits Per Sample = %d\n",bits_per_sample);
   printf(" Number of Channels = %d\n\n",num_ch);
 
 
   // reset output stream index 
   g_num_osamp = 0;
 
   // be polite 
   if(wbuff!=NULL) delete wbuff;
   if(wav!=NULL) delete wav;
   if(chk!=NULL) delete chk;
   fclose(fw);
   return 0;
 
/* int WAV_OUT::save_wave_file(char *fname) */}
 
 
 
 
 
/********************************************\
\********************************************/ 
WAV_OUT::WAV_OUT(double _fs_hz,int _bits_per_sample,int _num_ch)
{
 
   fs_hz = _fs_hz;
   bits_per_sample = _bits_per_sample;
   num_ch = _num_ch;
 
   g_wdata_out = NULL;
   g_num_osamp = 0;
   g_max_osamp = 0;
 
   return;
   
/* WAV_OUT::WAV_OUT(,,,) */}
 
 
 
 
 
/********************************************\
\********************************************/ 
WAV_OUT::WAV_OUT(WAV_IN *wav_in)
{
 
   fs_hz = wav_in->fs_hz;
   bits_per_sample = wav_in->bits_per_sample;
   num_ch = wav_in->num_ch;
 
   g_wdata_out = NULL;
   g_num_osamp = 0;
   g_max_osamp = 0;
 
   return;
   
/* WAV_OUT::WAV_OUT(,,,) */}
 
 
 
 
 
 
 
/********************************************\
\********************************************/ 
WAV_OUT::~WAV_OUT()
{ 
	if( g_wdata_out != NULL )
		delete[] g_wdata_out;
}
 
 

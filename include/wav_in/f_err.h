/****** NOTICE: LIMITATIONS ON USE AND DISTRIBUTION ********\
 
� This software is provided on an as-is basis, it may be 
� distributed in an unlimited fashion without charge provided
� that it is used for scholarly purposes - it is not for 
� commercial use or profit. This notice must remain unaltered. 
 
� Software by Dr Fred DePiero - CalPoly State University
 
\******************** END OF NOTICE ************************/ 

#ifndef FERR_H
#define FERR_H
 
#define thret_gerr(n,s) { printf("%s (%d)\n",s,n); exit(n); }
#define threx_gerr(n,s) { printf("%s (%d)\n",s,n); exit(n); }
#define threv_gerr(n,s) { printf("%s (%d)\n",s,n); exit(n); }
#define throw_gerr(n,s) { printf("%s (%d)\n",s,n); exit(n); }
#define reton_gerr()   
#define revon_gerr()   
#define clear_gerr()   
#define print_gerr()   
#define catch_gerr()   0   
 
#endif 
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define OK     0
#define ERROR -1
#define FAIL  -1
#define ON     1
#define OFF    0

#define MAX_IM_VAL          512
#define MAX_MED             100
#define MAX_STR_LEN	    100

using namespace std;

/* 

This script puts lead compensators on top of a patient+plastic tray egsphant. 

Written by Levi Burns, based on code written by Tony Teke

Last update: April 2018

This script just produces some numbers from the .egsphant instead of needing to open it, write down the numbers, and close again

 */


typedef struct
{
    int   x_num, y_num, z_num, num_mat;
    float x_bound[MAX_IM_VAL], y_bound[MAX_IM_VAL], z_bound[MAX_IM_VAL];
    float x_size, y_size, z_size;
    float x_start, y_start, z_start;
    // char med_name[MAX_STR_LEN][MAX_MED];
    char med_name[MAX_MED][MAX_STR_LEN]; // July 9, 2000: JVS:
    float estep[MAX_MED];
    int *mednum;
    float *densval;
} PHANT_STRUCT;



/* *********************************************************************** */
int readPhantomBoundaries(FILE *istrm, int nBounds, float *bounds)
{
  double tmpValue;

  for(int iBound=0; iBound < nBounds; iBound++) 
  {
    if(1!=fscanf(istrm,"%lf",&tmpValue)) {
      printf("\n ERROR: readPhantomBoundaries for boundary %d", iBound);
    }
    bounds[iBound] = (float) (tmpValue);       // read all the boundaries and store them into an array 
    //printf("%3.7f\t",bounds[iBound]);		//  that is in the PHANT_STRUCT

  }
  return(OK);
}
/* *********************************************************************** */
int read_phant(char *fname, PHANT_STRUCT *p)
{  

  FILE *fp;
  fp = fopen(fname,"r");
  if(fp == NULL)
  {
     printf("\n ERROR: opening file >%s",fname);return(FAIL);
  }
//TT
  fscanf(fp,"%d",&p->num_mat);		// egsphant files are text file see dosxyznrc for what it contains
  //printf("\n num_mat=%d",p->num_mat);
  int i,j,k;

  if(p->num_mat > MAX_MED)	// defined in phantomStructure.h and equal 100
  {
     printf("\n ERROR: %d Exceeds the maximum number of materials (%d)\n", p->num_mat, MAX_MED);
     return(FAIL);
  }

  for (i=0;i<p->num_mat;i++)
    if (fscanf(fp,"%s",p->med_name[i]) != 1)
    {
      printf("\n ERROR: fscan");
      return(FAIL);
    }
  //**************TT debug 
  //  for (i=0;i<p->num_mat;i++)
  //    printf("%s\n",p->med_name[i]);

  for (i=0;i<p->num_mat;i++)
    if (fscanf(fp,"%f",&p->estep[i]) != 1)
    {
      printf("\n ERROR: fscan");
      return(FAIL);
    }
  //**************TT debug 
  //for (i=0;i<p->num_mat;i++)
    //printf("%f\n",p->estep[i]);



  if (fscanf(fp,"%d%d%d",&p->x_num,&p->y_num,&p->z_num) !=3)  // read number of voxels in each direction X,Y,Z
  {
    printf("\n ERROR: fscan: reading in x_num,y_num,z_num");
    return(FAIL);
  }
  if( p->x_num > MAX_IM_VAL ||		// defined in phantomStructure.h and set to 255
      p->y_num > MAX_IM_VAL ||
      p->z_num > MAX_IM_VAL )
  {
     printf("\n ERROR: %d %d or %d Exceeds Maximum Number of Voxels (%d)", 
         p->x_num, p->y_num, p->z_num , MAX_IM_VAL);
     return(FAIL);
  }
  //printf("\n xvox=%d yvox=%d zvox=%d \n",p->x_num,p->y_num,p->z_num);

  // printf("\n Reading In Bounds\n");
  // Read in the x_bounds
   if(OK != readPhantomBoundaries(fp, p->x_num+1, p->x_bound) ) {
    printf("\n ERROR: Reading phantom x boundaries"); return(FAIL);
    } 

   printf("\n");
   // Read in the y_bounds
   if(OK != readPhantomBoundaries(fp, p->y_num+1, p->y_bound) ) {
    printf("\n ERROR: Reading phantom y boundaries"); return(FAIL);
    }

   printf("\n");
   // Read in the z_bounds
   if(OK != readPhantomBoundaries(fp, p->z_num+1, p->z_bound) ) {
    printf("\n ERROR: Reading phantom z boundaries"); return(FAIL);
    }
#ifdef DUMP_BOUNDS		// seems to be a special flag, not displayed during regular run
  printf ("\n After readPhantomBoundaries\n");
  for (i=0;i<p->x_num+1;i++) {
    fscanf(fp,"%f",&p->x_bound[i]);
    
    printf("%f\t",p->x_bound[i]);
    
    if(i%8 == 0) printf ("\n");
  }

  for (i=0;i<p->y_num+1;i++){
    fscanf(fp,"%f",&p->y_bound[i]);
    printf("%f\t",p->y_bound[i]);
    if(i%8 == 0) printf ("\n");
  }
  for (i=0;i<p->z_num+1;i++){
    fscanf(fp,"%f",&p->z_bound[i]);
    printf("%f\t",p->z_bound[i]);
    if(i%8 == 0) printf ("\n");
  }
#endif

  //printf("\n xmin: %f",&p->x_bound[0]);
  //printf("\n xmax: %f",&p->x_bound[x_num]);
  //printf("\n ymin: %f",&p->y_bound[0]);
  //printf("\n ymax: %f",&p->y_bound[y_num]);
  //printf("\n zmin: %f",&p->z_bound[0]);
  //printf("\n zmax: %f",&p->z_bound[z_num]);

  p->mednum = (int *)calloc(p->x_num*p->y_num*p->z_num,sizeof(int));
  if(p->mednum == NULL)
  {
     printf("\n ERROR: Allocating Memory for Int Array");
     printf("\n\t x %d y %d z %d", p->x_num,p->y_num,p->z_num);
     return(FAIL);
  }
  
  p->densval = (float *)calloc(p->x_num*p->y_num*p->z_num,sizeof(float));
  if(p->densval == NULL)
  {
     printf("\n ERROR: Allocating Memory for Float Array");
     return(FAIL);
  }
 
  printf("\n Reading In Medium Numbers\n");
  int nread = 0;


  for (k=0;k<p->z_num;k++) 
    {
      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
	   {
            if( fscanf(fp,"%1d",&p->mednum[k*p->x_num*p->y_num + j*p->x_num +i])== 1) 
		{ nread++;

		}

	   }   

	}
  
    }


  if(nread != p->z_num*p->y_num*p->x_num)
  {
    printf("\n ERROR: reading in mednum, read in %d, expected %d", nread,p->z_num*p->y_num*p->x_num);
  }


  // printf("\n Reading In Density Values\n");
  // nread = 0;

  // for(i=0;i<(p->x_num*p->y_num*p->z_num);i++)
  //	{
  //		fscanf(fp,"%f",&p->densval[i]);
  //		nread++;
  //	}


  //if(nread != p->z_num*p->y_num*p->x_num)
  //{
  //  printf("\n ERROR: reading in densval, read in %d, expected %d", nread,p->z_num*p->y_num*p->x_num);
  //}

  fclose(fp);
 // Some info to print out in terminal
  printf("\n For %s",fname);
  printf("\n Number of voxels %d %d %d",p->x_num,p->y_num,p->z_num);
  //printf("\n Size of voxels   %f %f %f",p->x_bound[1]-p->x_bound[0],p->y_bound[1]-p->y_bound[0],p->z_bound[1]-p->z_bound[0]);
  printf("\n Start of voxels  %f %f %f",p->x_bound[0],p->y_bound[0],p->z_bound[0]);
  printf("\n End of voxels  %f %f %f \n \n",p->x_bound[p->x_num],p->y_bound[p->y_num],p->z_bound[p->z_num]);
  return(OK);
}
/* *********************************************************************** */
/* *********************************************************************** */

/* *********************************************************************** */

int main(int argc, const char* argv[])
{

  //filenames: adjust the following lines of code if a new convention is desirable

//argv[1]= egsphant file name, no .egsphant

  char egsphantFileName2[256];


  
  strcat(egsphantFileName2,argv[1]);
  strcat(egsphantFileName2,".egsphant");



  // read First EGS4Phant ....

  PHANT_STRUCT Rando;

  // *****  Reading both Phantoms ************************

  printf("\n Loading information from %s", egsphantFileName2);
  if(read_phant(egsphantFileName2, &Rando) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Reading EGS4 Phantom File %s\n", egsphantFileName2); return(FAIL);
  }
  // *****************************************************


return(OK);
}

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

  // printf("\nreadPhantomBoundaries \n");
  for(int iBound=0; iBound < nBounds; iBound++) 
  {
    if(1!=fscanf(istrm,"%lf",&tmpValue)) {
      printf("\n ERROR: readPhantomBoundaries for boundary %d", iBound);
    }
    bounds[iBound] = (float) (tmpValue);       // read all the boundaries and store them into an array 
    //printf("%3.7f\t",bounds[iBound]);		//  that is in the PHANT_STRUCT
    //if(iBound%8 == 0) printf ("\n");
  }
  return(OK);
}
/* *********************************************************************** */
int readPhantomBoundariesExtended(FILE *istrm, int nBounds, float *bounds)
{
  double tmpValue;
  int extend_pixel=140;
  float pixel_size;
  // printf("\nreadPhantomBoundaries \n");
  for(int iBound=0+extend_pixel; iBound < nBounds+extend_pixel; iBound++) 
  {
    if(1!=fscanf(istrm,"%lf",&tmpValue)) {
      printf("\n ERROR: readPhantomBoundaries for boundary %d", iBound);
    }
    bounds[iBound] = (float) (tmpValue);       // read all the boundaries and store them into an array 
    //printf("%3.7f\t",bounds[iBound]);		//  that is in the PHANT_STRUCT
    //if(iBound%8 == 0) printf ("\n");
  }
  // need ta add  in the negative direction
  printf("\n ADDING THESE NEW COORDINATES \n");
  pixel_size=bounds[extend_pixel+2]-bounds[extend_pixel+1]; // pixel size will be a negative value in this case for Rando
  printf("\n Pixel_size = %f\n",pixel_size);
  printf("extend_pixel =%d\n",extend_pixel);
  for(int k=extend_pixel;k>=0;k--) {
    bounds[k]=bounds[k+1]-pixel_size;
    //printf("%3.7f\t",bounds[k]);
  }
  return(OK);
}
/* *********************************************************************** */
/* *********************************************************************** */
/* *********************************************************************** */
int write_phant(char *fname, PHANT_STRUCT *p)
{  
  FILE *fp;
  fp = fopen(fname,"w");
  if(fp == NULL)
    {
     printf("\n ERROR: opening file >%s",fname);
     return(FAIL);
    }
  else
     printf("\n Opened file %s",fname); 

  int i,j,k;
  fprintf(fp,"%d\n",p->num_mat);
  for (i=0;i<p->num_mat;i++)
    fprintf(fp,"%s\n",p->med_name[i]);

  for (i=0;i<p->num_mat;i++)
    fprintf(fp,"%g ",p->estep[i]);
  fprintf(fp,"\n");

  fprintf(fp,"%5d%5d%5d",p->x_num,p->y_num,p->z_num);

  for (i=0;i<p->x_num+1;i++){
    if(!(i%10)) fprintf(fp,"\n");
    fprintf(fp,"%9.7f ",p->x_bound[i]); // Used to be just %g format
  }
  // fprintf(fp,"\n");
  for (i=0;i<p->y_num+1;i++){
    if(!(i%10)) fprintf(fp,"\n");
    fprintf(fp,"%9.7f ",p->y_bound[i]);
  }
  // fprintf(fp,"\n");
  for (i=0;i<p->z_num+1;i++){
    if(!(i%10)) fprintf(fp,"\n");
    fprintf(fp,"%9.7f ",p->z_bound[i]);
  }
  fprintf(fp,"\n");

  for (k=0;k<p->z_num;k++) 
    {
      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
             //fprintf(fp,"%3d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]);
	     fprintf(fp,"%1d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]);
          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");
    }

  for (k=0;k<p->z_num;k++) 
    {
      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
             fprintf(fp,"%9.7f ",p->densval[k*p->x_num*p->y_num + j*p->x_num +i]);
          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");
    }

  fclose(fp);
  printf("\n For %s",fname);
  printf("\n Number of voxels %d %d %d",p->x_num,p->y_num,p->z_num);
  printf("\n Size of voxels   %f %f %f",p->x_bound[1]-p->x_bound[0],p->y_bound[1]-p->y_bound[0],p->z_bound[1]-p->z_bound[0]);
  printf("\n Start of voxels  %f %f %f",p->x_bound[0],p->y_bound[0],p->z_bound[0]);
  fflush(stdout);fflush(stderr);
  return(OK);
}
/* *********************************************************************** */
int read_phant_and_extend(char *fname, PHANT_STRUCT *p)  // need to include int gap_size as argument
{  
  printf("\n Reading In %s\n",fname);
  float gap_size =10.;
  FILE *fp;
  fp = fopen(fname,"r");
  if(fp == NULL)
  {
     printf("\n ERROR: opening file >%s",fname);return(FAIL);
  }
//TT
  fscanf(fp,"%d",&p->num_mat);		// egsphant files are text file see dosxyznrc for what it contains
  printf("\n num_mat=%d\n",p->num_mat);
  int i,j,k;
  /* if (fscanf(fp,"%d",&p->num_mat) != 1)			// Need to find where this is defined
  {
    printf("\n ERROR: fscan: num_mat"); return(FAIL);
  } */
  if(p->num_mat > MAX_MED)	// defined in phantomStructure.h and equal 100
  {
     printf("\n ERROR: %d Exceeds the maximum number of materials (%d)\n", p->num_mat, MAX_MED);
     return(FAIL);
  }

  // Scan material names
  for (i=0;i<p->num_mat;i++)
    if (fscanf(fp,"%s",p->med_name[i]) != 1)
    {
      printf("\n ERROR: fscan medium names");
      return(FAIL);
    }
  //**************TT debug 
  for (i=0;i<p->num_mat;i++)
    printf("%s\n",p->med_name[i]);

  // scan estep  obsolete data now but still needed
  for (i=0;i<p->num_mat;i++)
    if (fscanf(fp,"%f",&p->estep[i]) != 1)
    {
      printf("\n ERROR: fscan estep values");
      return(FAIL);
    }
  //**************TT debug 
  for (i=0;i<p->num_mat;i++)
    printf("%f\n",p->estep[i]);


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

  printf("\n xvox=%d yvox=%d zvox=%d \n",p->x_num,p->y_num,p->z_num);
  
  
  int extend_pixel=140;
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
   if(OK != readPhantomBoundariesExtended(fp, p->z_num+1, p->z_bound) ) {
    printf("\n ERROR: Reading phantom z boundaries"); return(FAIL);
    }

  

  p->mednum = (int *)calloc(p->x_num*p->y_num*(p->z_num+extend_pixel),sizeof(int));
  if(p->mednum == NULL)
  {
     printf("\n ERROR: Allocating Memory for Int Array");
     printf("\n\t x %d y %d z %d", p->x_num,p->y_num,p->z_num);
     return(FAIL);
  }
  
  p->densval = (float *)calloc(p->x_num*p->y_num*(p->z_num+extend_pixel),sizeof(float));
  if(p->densval == NULL)
  {
     printf("\n ERROR: Allocating Memory for Float Array");
     return(FAIL);
  }
 
  printf("\n Reading In Medium Numbers\n");
  int nread = 0, nread2 = 0;

  
  // need to change once for all the size of zvoxels before looping
  //p->z_num+=extend_pixel;
  
  // original phantom
  for (k=extend_pixel;k<p->z_num+extend_pixel;k++) 
    {
      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
	   {
	      if( fscanf(fp,"%1d",&p->mednum[k*p->x_num*p->y_num + j*p->x_num +i])== 1)
				{ nread++;
		  		//printf("%d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]); //TT debug
				}
	      //else  // TT debug
		//printf("ERROR READING MEDNUM i=%d j=%d k=%d \n",i,j,k);
	    } 
	}  
    }
  
  // now adding the extended part located at the begining
  for (k=0;k<extend_pixel;k++) 
    {
      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
	   {
	      p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]= 1;
	    } 
	}   
    }  

  // WILL NEED TO CHANGE THIS AND RE-ENABLE THIS FEATURE
  /*if((nread+nread2) != (p->z_num)*p->y_num*p->x_num)
  {
    printf("\n ERROR: reading in mednum, nread in %d, nread2 in %d, expected %d", nread,nread2,p->z_num*p->y_num*p->x_num);
  }*/

  // printf("\n Reading In Density Values\n");
  nread = 0;
  nread2=0;
  float air_density=0.0012048;
  for (k=extend_pixel;k<p->z_num+extend_pixel;k++) 
      for (j=0;j<p->y_num;j++) 
          for (i=0;i<p->x_num;i++) 
	  	{
            	 if(fscanf(fp,"%f",&p->densval[k*p->x_num*p->y_num + j*p->x_num +i])==1)
		    nread++;
		 //else   
		    //printf("ERROR READING DENSVAL AT i=%d j=%d k=%d \n",i,j,k); 
		}

	    // now adding the extended part located at the begining
  for (k=0;k<extend_pixel;k++) 
    {
      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
	   {
	      p->densval[k*p->x_num*p->y_num + j*p->x_num +i]= air_density;
	    } 
	}   
    }  
    
//TT modified debug
/*  for(i=0;i<(p->x_num*p->y_num*p->z_num);i++)
	{
		fscanf(fp,"%f",&p->densval[i]);
		nread++;
	}
 */

  if((nread+nread2) != p->z_num*p->y_num*p->x_num)
  {
    printf("\n ERROR: reading in densval, nread in %d, nread2 in %d, expected %d", nread,nread2,p->z_num*p->y_num*p->x_num);
  }

  p->z_num+=extend_pixel;
  
  fclose(fp);
  printf("\n For %s",fname);
  printf("\n Number of voxels %d %d %d",p->x_num,p->y_num,p->z_num);
  printf("\n Size of voxels   %f %f %f",p->x_bound[1]-p->x_bound[0],p->y_bound[1]-p->y_bound[0],p->z_bound[1]-p->z_bound[0]);
  printf("\n Start of voxels  %f %f %f",p->x_bound[0],p->y_bound[0],p->z_bound[0]);
  return(OK);
}

/* *********************************************************************** */

int main(int argc, const char* argv[])
{
//char egsphantFileName[MAX_STR_LEN];
//char* egsphantFileName="TBIFilter.egsphant";

//argv[1] = ID
//argv[2] = AP or PA

  char egsphantFileName[256];
  char outputegsphantFileName[256];

  strcpy(egsphantFileName,"lead_plastic_");
  strcat(egsphantFileName,argv[1]);
  strcat(egsphantFileName,"_");
  strcat(egsphantFileName,argv[2]);
  strcat(egsphantFileName,".egsphant");
  strcpy(outputegsphantFileName,"extended_lead_plastic_");
  strcat(outputegsphantFileName,argv[1]);
  strcat(outputegsphantFileName,"_");
  strcat(outputegsphantFileName,argv[2]);
  strcat(outputegsphantFileName,".egsphant");

//CHANGE NAMES HERE
//char *egsphantFileName="lead_plastic_VC_TBI00_APORPA.egsphant";
//char *outputegsphantFileName="extended_lead_VC_TBI00_APORPA.egsphant";
  // read First EGS4Phant ....
  PHANT_STRUCT phantom;			// created in phantomStructure.h
  printf("\n Loading information from %s", egsphantFileName);
  if(read_phant_and_extend(egsphantFileName, &phantom) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Reading EGS4 Phantom File %s\n", egsphantFileName); return(FAIL);
  }

  printf("\n Writting information into %s", outputegsphantFileName);
  if(write_phant(outputegsphantFileName, &phantom) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Writting EGS4 Phantom File %s\n", outputegsphantFileName); return(FAIL);
  }
 printf("\n");
return(OK);

}

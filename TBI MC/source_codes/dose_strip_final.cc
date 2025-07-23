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
    double *mednum;
    float *densval;
} PHANT_STRUCT;


// EDITED Levi Burns Aug 2 2017 - now a 3ddose strip file


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
int read_phant(char *fname, PHANT_STRUCT *p)
{  
    // printf("\n Reading In %s\n",fname);
  FILE *fp;
  fp = fopen(fname,"r");
  if(fp == NULL)
  {
     printf("\n ERROR: opening file >%s",fname);return(FAIL);
  }
//TT
  //LB fscanf(fp,"%d",&p->num_mat);		// egsphant files are text file see dosxyznrc for what it contains
  // printf("\n num_mat=%d",p->num_mat);
  int i,j,k;



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
    //    fscanf(fp,"%f",&p->x_bound[i]);
    printf("%f\t",p->x_bound[i]);
    if(i%8 == 0) printf ("\n");
  }

  for (i=0;i<p->y_num+1;i++){
    // fscanf(fp,"%f",&p->y_bound[i]);
    printf("%f\t",p->y_bound[i]);
    if(i%8 == 0) printf ("\n");
  }
  for (i=0;i<p->z_num+1;i++){
    // fscanf(fp,"%f",&p->z_bound[i]);
    printf("%f\t",p->z_bound[i]);
    if(i%8 == 0) printf ("\n");
  }
#endif

  p->mednum = (double *)calloc(p->x_num*p->y_num*p->z_num,sizeof(double));
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
	     if( fscanf(fp,"%lf",&p->mednum[k*p->x_num*p->y_num + j*p->x_num +i])== 1) //LB changed string format specifier
		{ nread++;

		}
	    else  // TT debug
		printf("i=%d j=%d k=%d",i,j,k);
	   }   

	}
  
    }


  if(nread != p->z_num*p->y_num*p->x_num)
  {
    printf("\n ERROR: reading in mednum, read in %d, expected %d", nread,p->z_num*p->y_num*p->x_num);
  }


  // printf("\n Reading In Density Values\n");
  nread = 0;

  for(i=0;i<(p->x_num*p->y_num*p->z_num);i++)
	{
		fscanf(fp,"%f",&p->densval[i]);
		nread++;
	}


  if(nread != p->z_num*p->y_num*p->x_num)
  {
    printf("\n ERROR: reading in densval, read in %d, expected %d", nread,p->z_num*p->y_num*p->x_num);
  }

  fclose(fp);
  printf("\n For %s",fname);
  printf("\n Number of voxels %d %d %d",p->x_num,p->y_num,p->z_num);
  printf("\n Size of voxels   %f %f %f",p->x_bound[1]-p->x_bound[0],p->y_bound[1]-p->y_bound[0],p->z_bound[1]-p->z_bound[0]);
  printf("\n Start of voxels  %f %f %f",p->x_bound[0],p->y_bound[0],p->z_bound[0]);
  return(OK);
}
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

  int extend_pixel = 140;
  int extend_pixel_front = 0; //LB Sept 18 2017 to deal with front extension

  // fprintf(fp,"%d\n",p->num_mat);
  //for (i=0;i<p->num_mat;i++)
  //  fprintf(fp,"%s\n",p->med_name[i]);

  //for (i=0;i<p->num_mat;i++)
  //  fprintf(fp,"%.0g ",p->estep[i]);
  //fprintf(fp,"\n");

  //fprintf(fp,"%5d%5d%5d",p->x_num,p->y_num,p->z_num)
  fprintf(fp,"%5d%5d%5d",p->x_num,(p->y_num)-28,(p->z_num)-extend_pixel-extend_pixel_front); //LB CHANGE
  fprintf(fp,"\n");

  for (i=0;i<p->x_num+1;i++){    // LB DO NOT CHANGE

    fprintf(fp,"%7.5f ",p->x_bound[i]);

  }
   fprintf(fp,"\n");

  for (i=28;i<p->y_num+1;i++){    //LB CHANGE 
    // if(!(i%10)) fprintf(fp,"\n");
    fprintf(fp,"%7.5f ",p->y_bound[i]);
  }
   fprintf(fp,"\n");

  for (i=extend_pixel;i<p->z_num+1-extend_pixel_front;i++){     //LB CHANGE 
    //  if(!(i%10)) fprintf(fp,"\n");  
      fprintf(fp,"%7.5f ",p->z_bound[i]);

  }
  fprintf(fp,"\n");


  // LB ADJUST INCREMENTING
  //DOSE_VALS
  for (k=extend_pixel;k<p->z_num-extend_pixel_front;k++) //LB: made each z-slice twice
    {
      for (j=28;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 


	   {fprintf(fp,"%9.7E ",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]);
	   }


          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");


    }

      
  //LB ADJUST INCREMENTING

  //UNCERTS
  for (k=extend_pixel;k<p->z_num-extend_pixel_front;k++) //LB: made each z-slice twice
    {
      for (j=28;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 

	    {fprintf(fp,"%.8f ",p->densval[k*p->x_num*p->y_num + j*p->x_num +i]);
	      // fprintf(fp,"%.8e ",p->densval[k*p->x_num*p->y_num + j*p->x_num +i]); }
	    }
          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");




    }

  fclose(fp);
  printf("\n For %s",fname);
  printf("\n Number of voxels %d %d %d",p->x_num,(p->y_num)-28,(p->z_num)-extend_pixel-extend_pixel_front); //reschange
  printf("\n Size of voxels   %f %f %f",p->x_bound[1]-p->x_bound[0],p->y_bound[1]-p->y_bound[0],p->z_bound[1]-p->z_bound[0]); //reschange - this is just screen output and I tried to change it but maybe it's still wrong, whatever
  printf("\n Start of voxels  %f %f %f",p->x_bound[0],p->y_bound[0],p->z_bound[0]);
  fflush(stdout);fflush(stderr);
  return(OK);
}
/* *********************************************************************** */

/* *********************************************************************** */
//	This code will fuse 2 pb to Rando without any shift in x direction
//	because it is not needed for now
/* *********************************************************************** */
int main(int argc, const char* argv[])
{

  char egsphantFileName1[256];
  char outputegsphantFileName[256];

  strcpy(egsphantFileName1,argv[1]);
  strcat(egsphantFileName1,".3ddose");
  strcpy(outputegsphantFileName,argv[1]);
  strcat(outputegsphantFileName,"_trimmed.3ddose");

//***********************************
// 	CHANGE FILENAMES HERE
//***********************************
//char* egsphantFileName1="SupineOrProne_VC_TBI00.3ddose";
//char* outputegsphantFileName="SupineOrProne_VC_TBI00_trimmed.3ddose";

// CHANGE PIXEL RESOLUTION HERE IF NEEDED
//float pixel_size=0.5;  // 2.5 mm fixed by hand but will need to change this
//float air_density=0.0012048;

  // read First EGS4Phant ....
  PHANT_STRUCT TBIFilter;
  //  PHANT_STRUCT pb_Rando;

  // *****  Reading both Phantoms ************************
  printf("\n Loading information from %s", egsphantFileName1);
  if(read_phant(egsphantFileName1, &TBIFilter) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Reading EGS4 Phantom File %s\n", egsphantFileName1); return(FAIL);
  }


  
  float y_res=TBIFilter.y_bound[2]-TBIFilter.y_bound[1]; // maybe not needed

  
  printf("\n y_res= %f \n",y_res);

    
  
  printf("\n Writting information into %s", outputegsphantFileName);
  if(write_phant(outputegsphantFileName, &TBIFilter) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Writting EGS4 Phantom File %s\n", outputegsphantFileName); return(FAIL);
  } 
  printf("\n");
return(OK);
}

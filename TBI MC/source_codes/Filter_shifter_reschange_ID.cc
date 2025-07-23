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
  fscanf(fp,"%d",&p->num_mat);		// egsphant files are text file see dosxyznrc for what it contains
  printf("\n num_mat=%d",p->num_mat);
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

  for (i=0;i<p->num_mat;i++)
    if (fscanf(fp,"%s",p->med_name[i]) != 1)
    {
      printf("\n ERROR: fscan");
      return(FAIL);
    }
  //**************TT debug 
  for (i=0;i<p->num_mat;i++)
    printf("%s\n",p->med_name[i]);

  for (i=0;i<p->num_mat;i++)
    if (fscanf(fp,"%f",&p->estep[i]) != 1)
    {
      printf("\n ERROR: fscan");
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
		  //printf("%d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]); //TT debug
		}
	    else  // TT debug
		printf("i=%d j=%d k=%d",i,j,k);
	   }   
	//printf("\n");
	}
      //printf("\n");   
    }
/*
//TT modified debug
  for(i=0;i<(p->x_num*p->y_num*p->z_num);i++)
	{
		fscanf(fp,"%3d",&p->mednum[i]);
		nread++;
	} */

  if(nread != p->z_num*p->y_num*p->x_num)
  {
    printf("\n ERROR: reading in mednum, read in %d, expected %d", nread,p->z_num*p->y_num*p->x_num);
  }


  // printf("\n Reading In Density Values\n");
  nread = 0;
/*
  for (k=0;k<p->z_num;k++) 
      for (j=0;j<p->y_num;j++) 
          for (i=0;i<p->x_num;i++) 
	  {
             if(fscanf(fp,"%f",&p->densval[k*p->x_num*p->y_num + j*p->x_num +i])==1) nread++;
	  }  */
//TT modified debug
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
  fprintf(fp,"%d\n",p->num_mat);
  for (i=0;i<p->num_mat;i++)
    fprintf(fp,"%s\n",p->med_name[i]);

  for (i=0;i<p->num_mat;i++)
    fprintf(fp,"%.0g ",p->estep[i]);
  fprintf(fp,"\n");

  //fprintf(fp,"%5d%5d%5d",p->x_num,p->y_num,p->z_num)
  fprintf(fp,"%5d%5d%5d",2*(p->x_num),p->y_num,2*(p->z_num)); //reschange

  for (i=0;i<p->x_num+1;i++){
    // if(!(i%10)) fprintf(fp,"\n");
    if(!(i%5)) fprintf(fp,"\n");
    fprintf(fp,"%9.7f ",p->x_bound[i]);
    if(i!=p->x_num)
      fprintf(fp,"%9.7f ",0.25+p->x_bound[i]);  //reschange
 // Used to be just %g format
  }
  // fprintf(fp,"\n");

  for (i=0;i<p->y_num+1;i++){
    if(!(i%10)) fprintf(fp,"\n");
    fprintf(fp,"%9.7f ",p->y_bound[i]);
  }
  // fprintf(fp,"\n");

    // if(!(i%10)) fprintf(fp,"\n");
    // if(!(i%5)) fprintf(fp,"\n");  for (i=0;i<p->z_num+1;i++){
    for (i=0;i<p->z_num+1;i++){ 
      if(!(i%5)) fprintf(fp,"\n");  
      fprintf(fp,"%9.7f ",p->z_bound[i]);
      if(i!=p->z_num)
        fprintf(fp,"%9.7f ",0.25+p->z_bound[i]);  //reschange
  }
  fprintf(fp,"\n");

  //MEDNUM
  for (k=0;k<p->z_num;k++) //LB: made each z-slice twice
    {

      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
	    //reschange; now each x value is printed twice
	     {fprintf(fp,"%1d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]);
	       fprintf(fp,"%1d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]);}

          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");


      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
	    //reschange; now each x value is printed twice
	     {fprintf(fp,"%1d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]);
	       fprintf(fp,"%1d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]);}

          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");

    }


  //DENSVAL
  for (k=0;k<p->z_num;k++) //LB: made each z-slice twice
    {
      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 

	    {fprintf(fp,"%.8e ",p->densval[k*p->x_num*p->y_num + j*p->x_num +i]);
	     fprintf(fp,"%.8e ",p->densval[k*p->x_num*p->y_num + j*p->x_num +i]); }

          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");



      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 

	    {fprintf(fp,"%.8e ",p->densval[k*p->x_num*p->y_num + j*p->x_num +i]);
	     fprintf(fp,"%.8e ",p->densval[k*p->x_num*p->y_num + j*p->x_num +i]); }

          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");


    }

  fclose(fp);
  printf("\n For %s",fname);
  printf("\n Number of voxels %d %d %d",2*(p->x_num),p->y_num,2*(p->z_num)); //reschange
  //  printf("\n Size of voxels   %f %f %f",(p->x_bound[1]-p->x_bound[0])/2.0,p->y_bound[1]-p->y_bound[0],(p->z_bound[1]-p->z_bound[0]))/2.0; //reschange
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

  //argv[1] = ID (two digits)
  //argv[2] = AP or PA
  //argv[3] = Tx_x
  //argv[4] = Tx_z
  //argv[5] = SSD
  //argv[6] = Tx_y


  char outputegsphantFileName[256];
  strcpy(outputegsphantFileName,"shiftedfilter_");
  strcat(outputegsphantFileName,argv[1]);
  strcat(outputegsphantFileName,"_");
  strcat(outputegsphantFileName,argv[2]);
  strcat(outputegsphantFileName,".egsphant");

//***********************************
// 	CHANGE FILENAMES HERE
//***********************************
char* egsphantFileName1="Filter_PMMA_material.egsphant";
//char* outputegsphantFileName="shiftedfilter_VC_TBI00_APORPA.egsphant";

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

  // *******************************************************
  // 	Shifting z axis of pb by z_shift
  // CHANGE COORDINATE OF BBs HERE IF NEEDED
  // TO DO: NEED TO INTRODUCE X SHIFT AS WELL AT SOME POINT
  //********************************************************
  float z_shift;
  float Tx_z=atof(argv[4]);  // Treatment centre z coordinate where the BB of filter should be aligned with
  
  float Filter_x=-0.1;
  float x_shift;
  float Tx_x=atof(argv[3]);
  x_shift= Tx_x-Filter_x;
  
  // Don't change the Filter_z
  float Filter_z=-101.0; // Coordinate of BB in z direction located in the centre of the filter
  z_shift = Tx_z-Filter_z; //-26.75; // calculated manually for this specific case

  printf("\n x_shift=%f \n",x_shift);

  for(int i=0;i<TBIFilter.x_num+1;i++)    // +1 because boundaries
	TBIFilter.x_bound[i] += x_shift;

  
  printf("\n z_shift=%f \n",z_shift);

  for(int i=0;i<TBIFilter.z_num+1;i++)    // +1 because boundaries
	TBIFilter.z_bound[i] += z_shift;
/*  
  //TT debug
  printf("\nZ direction boundaries TBIFilter\n");
  for(int i=0;i<TBIFilter.z_num+1;i++)
	printf("%f ",TBIFilter.z_bound[i]);
*/ 
//float y_shift;
//float Couch_surface=-18.30;
//float dist_bottom_filter_from_couch=131.3;

  float Tx_y=atof(argv[6]);
  float SSD=atof(argv[5]);

//y_shift=-(Couch_surface-dist_bottom_filter_from_couch) +TBIFilter.y_bound[TBIFilter.y_num-2];

//  for(int i=0;i<TBIFilter.y_num+1;i++)    // +1 because boundaries
//	TBIFilter.y_bound[i] -= y_shift;//-140;

  
  float y_res=TBIFilter.y_bound[2]-TBIFilter.y_bound[1]; // maybe not needed
  float y_filter_bottom=Tx_y-SSD+56.5+2*y_res; // WARNING must be the second index of the coordinate because second slice
  
  printf("\n y_res= %f \n",y_res);
  // fprintf(fp,"\n");
  for (int i=0;i<TBIFilter.y_num+1;i++){
    //if(!(i%10)) fprintf(fp,"\n");
    //fprintf(fp,"%9.7f ",y_filter_bottom -i*y_res); //p->y_bound[i]);
    TBIFilter.y_bound[i]=y_filter_bottom -(TBIFilter.y_num-i)*y_res;
  }
    
  
  printf("\n Writing information into %s", outputegsphantFileName);
  if(write_phant(outputegsphantFileName, &TBIFilter) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Writting EGS4 Phantom File %s\n", outputegsphantFileName); return(FAIL);
  } 
  printf("\n");
return(OK);
}

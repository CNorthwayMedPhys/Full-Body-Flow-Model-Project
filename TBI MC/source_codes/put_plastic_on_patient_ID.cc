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

This script puts a plastic tray on top of a patient egsphant. 

Written by Levi Burns, based on code written by Tony Teke

Last update: April 2018

Updated July 2018 to stop using two-digit IDs

INSTRUCTIONS: turn this script into an executable (e.g. g++ -Wall put_plastic_on_patient -o plastic), then with the executable, enter four arguments:


./plastic originalID newID ("AP or "PA") shift_y

examples:
./plastic 10051870 B5 AP -22.0
./plastic 1224849 01 PA -23.0

newID is used because I was anonymizing the patient data. 

Files that must be in the same folder:
plastic3.egsphant (an egsphant produced in Matlab for the plastic tray)
(patientID)_(APorPA).egsphant, e.g. 1224849_AP.egsphant (produced already in the TBI raytracing program)

Output file will be: plastic_VC_TBI(newID)_(APorPA).egsphant, e.g. plastic_VC_TBI01_PA.egsphant

Shift_y calculation: (y_bed) - (44.8-n(3.7)) where n is the number of styro
 y_bed is the bed y-coordinate, obtained from the egsphant in dosxyzshow

 */


typedef struct
{
    int   x_num, y_num, z_num, num_mat;
    float x_bound[MAX_IM_VAL], y_bound[MAX_IM_VAL], z_bound[MAX_IM_VAL];
    float x_size, y_size, z_size;
    float x_start, y_start, z_start;
    char med_name[MAX_MED][MAX_STR_LEN]; 
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
    //if(iBound%8 == 0) printf ("\n");
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

  fscanf(fp,"%d",&p->num_mat);		// egsphant files are text file see dosxyznrc for what it contains
  printf("\n num_mat=%d",p->num_mat);
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

  // Reading in bounds;
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

		}

	   }   

	}
  
    }


  if(nread != p->z_num*p->y_num*p->x_num)
  {
    printf("\n ERROR: reading in mednum, read in %d, expected %d", nread,p->z_num*p->y_num*p->x_num);
  }


  // Reading In Density Values;
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
  // Some info to print out in terminal
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

  //Writing into the egsphant things that are needed in that format

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


  //write in medium numbers and density values

  for (k=0;k<p->z_num;k++) 
    {
      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
             
	     if(p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]!=0)
		fprintf(fp,"%1d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]);
	     else {

		fprintf(fp,"1");
	     }
          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");
    }

float air_density=0.0012048;

  for (k=0;k<p->z_num;k++) 
    {
      for (j=0;j<p->y_num;j++) 
	{
          for (i=0;i<p->x_num;i++) 
            if( p->densval[k*p->x_num*p->y_num + j*p->x_num +i] != 0.)
	      fprintf(fp,"%9.7f ",p->densval[k*p->x_num*p->y_num + j*p->x_num +i]);
	    else {
	      fprintf(fp,"%9.7f ",air_density); 

	    }
          fprintf(fp,"\n");
	}
      fprintf(fp,"\n");
    }

  fclose(fp);
  printf("\n For %s",fname);
  //Output to the terminal
  printf("\n Number of voxels %d %d %d",p->x_num,p->y_num,p->z_num);
  printf("\n Size of voxels   %f %f %f",p->x_bound[1]-p->x_bound[0],p->y_bound[1]-p->y_bound[0],p->z_bound[1]-p->z_bound[0]);
  printf("\n Start of voxels  %f %f %f",p->x_bound[0],p->y_bound[0],p->z_bound[0]);
  fflush(stdout);fflush(stderr);
  return(OK);
}
/* *********************************************************************** */


/* *********************************************************************** */

int main(int argc, const char* argv[])
{

  //filenames: adjust the following lines of code if a new convention is desirable

//argv[1] = originalID
//argv[2] = "AP" or "PA"
//argv[3] = shift_y for plastic, entered as a negative value

  char egsphantFileName2[256];
  char outputegsphantFileName[256];

  strcpy(egsphantFileName2,argv[1]);
  strcat(egsphantFileName2,"_");
  strcat(egsphantFileName2,argv[2]);
  strcat(egsphantFileName2,".egsphant");
  strcpy(outputegsphantFileName,"plastic_");
  strcat(outputegsphantFileName,argv[1]);
  strcat(outputegsphantFileName,"_");
  strcat(outputegsphantFileName,argv[2]);
  strcat(outputegsphantFileName,".egsphant");

char* egsphantFileName1="plastic3.egsphant";



//float pixel_size=0.5;  // 2.5 mm fixed by hand but will need to change this
float air_density=0.0012048;

  // read First EGS4Phant ....
  PHANT_STRUCT pb;
  PHANT_STRUCT Rando;

  // *****  Reading both Phantoms ************************
  printf("\n Loading information from %s", egsphantFileName1);
  if(read_phant(egsphantFileName1, &pb) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Reading EGS4 Phantom File %s\n", egsphantFileName1); return(FAIL);
  }

  printf("\n Loading information from %s", egsphantFileName2);
  if(read_phant(egsphantFileName2, &Rando) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Reading EGS4 Phantom File %s\n", egsphantFileName2); return(FAIL);
  }
  // *****************************************************

  // *********************************************************
  //   Now fusing both phantoms into a new one
  PHANT_STRUCT pb_Rando;

  pb_Rando.num_mat=pb.num_mat+Rando.num_mat;
// attach names of material
for(int i=0;i<Rando.num_mat;i++)
  for(int j=0;j<20;j++)
    pb_Rando.med_name[i][j]=Rando.med_name[i][j];
  
for(int i=0;i<20;i++)
  pb_Rando.med_name[pb_Rando.num_mat-1][i]=pb.med_name[0][i];

  // first deal with dimensions
  pb_Rando.x_num=Rando.x_num;
  pb_Rando.y_num=Rando.y_num+pb.y_num+1;
  pb_Rando.z_num=Rando.z_num;

  //    dealing with the boundaries
  //*********************
  //Shift_y calculation: (y_bed) - (44.8-n(3.7)) where n is the number of styro
  // y_bed is the bed y-coordinate, obtained from the egsphant in dosxyzshow
  //*********************
  float shift_y=atof(argv[3]); // Hand calc where to shift the plastic
  
  for(int i=0;i<Rando.x_num+1;i++)
	pb_Rando.x_bound[i]=Rando.x_bound[i];
  for(int i=0;i<Rando.z_num+1;i++)
	pb_Rando.z_bound[i]=Rando.z_bound[i];
  for(int i=0;i<pb.y_num+1;i++)
	pb_Rando.y_bound[i]=pb.y_bound[i]+shift_y-pb.y_bound[1];
  
  printf("\ny_bound_pb %f %f \n",pb.y_bound[0],pb.y_bound[1]);

  // adding the Rando part
  for(int i=0;i<Rando.y_num+1;i++)
		pb_Rando.y_bound[i+pb.y_num+1]=Rando.y_bound[i];

//float estep[MAX_MED];
  for(int l=0;l<pb_Rando.num_mat;l++)
	pb_Rando.estep[l]=1.;
  // TT debug
  //for(int i=0;i<Rando.y_num+pb.y_num+2;i++)
  //	printf(" %f ",pb_Rando.y_bound[i]);

  //   now fusing material numbers
  //int counter=0;
//int aux=1;

  pb_Rando.mednum = (int *)calloc(pb_Rando.x_num*pb_Rando.y_num*pb_Rando.z_num,sizeof(int));
  if(pb_Rando.mednum == NULL)
  {
     printf("\n ERROR: Allocating Memory for Int Array");
     printf("\n\t x %d y %d z %d", pb_Rando.x_num,pb_Rando.y_num,pb_Rando.z_num);
     return(FAIL);
  }
printf("pb.x_num=%d\n",pb.x_num);
printf("pb.y_num=%d\n",pb.y_num);
printf("pb.z_num=%d\n",pb.z_num);

printf("pb_Rando.x_num=%d\n",pb_Rando.x_num);
printf("pb_Rando.y_num=%d\n",pb_Rando.y_num);
printf("pb_Rando.z_num=%d\n",pb_Rando.z_num);
//int zminpad=0;

//printf("\n counter=%d\n",counter);


//TT Debug?
// pb part with air gap
  for(int k=0;k<pb_Rando.z_num;k++)//k<pb_Rando.z_num;k++)
//int k=0;
	for(int j=0;j<pb.y_num+1;j++)
		for(int i=0;i<pb_Rando.x_num;i++)
		{
			//printf(" i=%d j=%d k=%d \n",i,j,k);
			if( j == pb.y_num)  // fine because j -> Rando.y_num+1 in the for loop
				pb_Rando.mednum[k*pb_Rando.x_num*pb_Rando.y_num + j*pb_Rando.x_num +i] = 1;
			else  
				pb_Rando.mednum[k*pb_Rando.x_num*pb_Rando.y_num + j*pb_Rando.x_num +i] =  pb.mednum[k*pb.x_num*pb.y_num + j*pb.x_num +i];
		}
		

// Rando part
  for(int k=0;k<pb_Rando.z_num;k++)//k<pb_Rando.z_num;k++)
//int k=0;
	for(int j=0;j<Rando.y_num;j++)
		for(int i=0;i<pb.x_num;i++)
		{ 
				pb_Rando.mednum[k*pb_Rando.x_num*pb_Rando.y_num + (j+pb.y_num+1)*pb_Rando.x_num +i] =  Rando.mednum[k*Rando.x_num*Rando.y_num + j*Rando.x_num +i];
		}



  // printf("counter=%d\n",counter);

  pb_Rando.densval = (float *)calloc(pb_Rando.x_num*pb_Rando.y_num*pb_Rando.z_num,sizeof(float));
  if(pb_Rando.densval == NULL)
  {
     printf("\n ERROR: Allocating Memory for Float Array");
     return(FAIL);
  }


//TT Debug?
// pb part with air gap
  for(int k=0;k<pb_Rando.z_num;k++)//k<pb_Rando.z_num;k++)
//int k=0;
	for(int j=0;j<pb.y_num+1;j++)
		for(int i=0;i<pb_Rando.x_num;i++)
		{
			//printf(" i=%d j=%d k=%d \n",i,j,k);
			if( j == pb.y_num)  // fine because j -> Rando.y_num+1 in the for loop
				pb_Rando.densval[k*pb_Rando.x_num*pb_Rando.y_num + j*pb_Rando.x_num +i] = air_density;
			else  
				pb_Rando.densval[k*pb_Rando.x_num*pb_Rando.y_num + j*pb_Rando.x_num +i] =  pb.densval[k*pb.x_num*pb.y_num + j*pb.x_num +i];
		}
		

// Rando part
  for(int k=0;k<pb_Rando.z_num;k++)//k<pb_Rando.z_num;k++)
//int k=0;
	for(int j=0;j<Rando.y_num;j++)
		for(int i=0;i<pb.x_num;i++)
		{ 
				pb_Rando.densval[k*pb_Rando.x_num*pb_Rando.y_num + (j+pb.y_num+1)*pb_Rando.x_num +i] =  Rando.densval[k*Rando.x_num*Rando.y_num + j*Rando.x_num +i];
		}





  printf("\n Writing information into %s", outputegsphantFileName);
  if(write_phant(outputegsphantFileName, &pb_Rando) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Writing EGS4 Phantom File %s\n", outputegsphantFileName); return(FAIL);
  } 
  printf("\n");

return(OK);
}

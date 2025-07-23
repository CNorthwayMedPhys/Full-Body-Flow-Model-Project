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

//argv[1] = ID
//argv[2] = AP or PA
//argv[3] = Supine or Prone

 */



typedef struct
{
    int   x_num, y_num, z_num, num_mat;
    float x_bound[MAX_IM_VAL], y_bound[MAX_IM_VAL], z_bound[MAX_IM_VAL];
    float x_size, y_size, z_size;
    float x_start, y_start, z_start;
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

		}
	    else  // TT debug
		printf("i=%d j=%d k=%d",i,j,k);
	   }   
	//printf("\n");
	}
      //printf("\n");   
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
	    if(p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]==10) fprintf(fp,"%1d",8);  //hopefully all my phantoms have 10 mats...
	   else fprintf(fp,"%1d",p->mednum[k*p->x_num*p->y_num + j*p->x_num +i]);
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


/* *********************************************************************** */
int main(int argc, const char* argv[])
{

  //argv[1] = ID
  //argv[2] = AP or PA


  char egsphantFileName1[256];
  char egsphantFileName2[256];
  char outputegsphantFileName[256];

  strcpy(egsphantFileName1,"shiftedfilter_");
  strcat(egsphantFileName1,argv[1]);
  strcat(egsphantFileName1,"_");
  strcat(egsphantFileName1,argv[2]);
  strcat(egsphantFileName1,".egsphant");
  strcpy(egsphantFileName2,"extended_lead_plastic_");
  strcat(egsphantFileName2,argv[1]);
  strcat(egsphantFileName2,"_");
  strcat(egsphantFileName2,argv[2]);
  strcat(egsphantFileName2,".egsphant");
  strcpy(outputegsphantFileName,argv[1]);
  strcat(outputegsphantFileName,"_");
  strcat(outputegsphantFileName,argv[2]);
  strcat(outputegsphantFileName,"_complete");
  strcat(outputegsphantFileName,".egsphant");

float pixel_size=0.5;  // 2.5 mm fixed by hand but will need to change this
float air_density=0.0012048;

  // read First EGS4Phant ....
  PHANT_STRUCT TBIFilter;
  PHANT_STRUCT pb_Rando;

  // *****  Reading both Phantoms ************************
  printf("\n Loading information from %s", egsphantFileName1);
  if(read_phant(egsphantFileName1, &TBIFilter) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Reading EGS4 Phantom File %s\n", egsphantFileName1); return(FAIL);
  }

  printf("\n Loading information from %s", egsphantFileName2);
  if(read_phant(egsphantFileName2, &pb_Rando) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Reading EGS4 Phantom File %s\n", egsphantFileName2); return(FAIL);
  }
  // *****************************************************

  float TBIFilter_z_min=0., TBIFilter_z_max=0., pb_Rando_z_min=0., pb_Rando_z_max=0.;
  float TBIFilter_x_min=0., TBIFilter_x_max=0., pb_Rando_x_min=0., pb_Rando_x_max=0.;

  TBIFilter_z_min=TBIFilter.z_bound[0];
  TBIFilter_z_max=TBIFilter.z_bound[TBIFilter.z_num];

  pb_Rando_z_min=pb_Rando.z_bound[0];
  pb_Rando_z_max=pb_Rando.z_bound[pb_Rando.z_num];

  TBIFilter_x_min=TBIFilter.x_bound[0];
  TBIFilter_x_max=TBIFilter.x_bound[TBIFilter.x_num];

  pb_Rando_x_min=pb_Rando.x_bound[0];
  pb_Rando_x_max=pb_Rando.x_bound[pb_Rando.x_num];


  printf("\n pb_Rando_z_min=%f \t pb_Rando_z_max=%f \t TBI_z_min=%f \t TBI_z_max=%f \n",pb_Rando_z_min,pb_Rando_z_max,TBIFilter_z_min,TBIFilter_z_max);

  printf("\n pb_Rando_x_min=%f \t pb_Rando_x_max=%f \t TBI_x_min=%f \t TBI_x_max=%f \n",pb_Rando_x_min,pb_Rando_x_max,TBIFilter_x_min,TBIFilter_x_max);


  int zminpad=0,zmaxpad=0,xminpad=0,xmaxpad=0;

  zminpad=abs(static_cast<int>((pb_Rando_z_min-TBIFilter_z_min)/pixel_size));
  zmaxpad=abs(static_cast<int>((pb_Rando_z_max-TBIFilter_z_max)/pixel_size));

  xminpad=abs(static_cast<int>((pb_Rando_x_min-TBIFilter_x_min)/pixel_size));
  xmaxpad=abs(static_cast<int>((pb_Rando_x_max-TBIFilter_x_max)/pixel_size));

  printf(" zminpad=%d \t zmaxpad=%d \n",zminpad,zmaxpad);
  printf(" xminpad=%d \t xmaxpad=%d \n",xminpad,xmaxpad);

  // *********************************************************
  //   Now fusing both phantoms into a new one
  PHANT_STRUCT TBI;

  TBI.num_mat=pb_Rando.num_mat+1;
for(int i=0;i<pb_Rando.num_mat;i++)
  for(int j=0;j<20;j++)
  TBI.med_name[i][j]=pb_Rando.med_name[i][j];

for(int i=0;i<20;i++)
  TBI.med_name[TBI.num_mat-1][i]=TBIFilter.med_name[1][i];

  // first deal with dimensions
  TBI.x_num=pb_Rando.x_num;
  TBI.y_num=pb_Rando.y_num+TBIFilter.y_num+1;
  TBI.z_num=pb_Rando.z_num;

  //    dealing with the boundaries
  for(int i=0;i<pb_Rando.x_num+1;i++)
	TBI.x_bound[i]=pb_Rando.x_bound[i];
  for(int i=0;i<pb_Rando.z_num+1;i++)
	TBI.z_bound[i]=pb_Rando.z_bound[i];

  for(int i=0;i<TBIFilter.y_num+1;i++)
	TBI.y_bound[i]=TBIFilter.y_bound[i];

  // adding the Rando part
  for(int i=0;i<pb_Rando.y_num+1;i++)
		TBI.y_bound[i+TBIFilter.y_num+1]=pb_Rando.y_bound[i];
  // TT debug
  //for(int i=0;i<TBI.y_num+1;i++)
  //	printf(" %f ",TBI.y_bound[i]);

  //   now fusing material numbers
int counter=0;
//int aux=1;

  TBI.mednum = (int *)calloc(TBI.x_num*TBI.y_num*TBI.z_num,sizeof(int));
  if(pb_Rando.mednum == NULL)
  {
     printf("\n ERROR: Allocating Memory for Int Array");
     printf("\n\t x %d y %d z %d", TBI.x_num,TBI.y_num,TBI.z_num);
     return(FAIL);
  }
//printf("TBI.z_num=%d\n",TBI.z_num);

//printf("\n counter=%d\n",counter);
  printf("fusing med_num\n");
  for(int k=0;k<TBI.z_num;k++)//k<pb_Rando.z_num;k++)
//int k=0;
	for(int j=0;j<TBI.y_num;j++)
		for(int i=0;i<TBI.x_num;i++)
		{
			//printf(" i=%d j=%d k=%d \n",i,j,k);
			if( j == TBIFilter.y_num)
				//pb_Rando.mednum[k*pb_Rando.x_num*pb_Rando.y_num + j*pb_Rando.x_num +i] = 1;
				TBI.mednum[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] = 1;

			else if ( ((j<TBIFilter.y_num) && (k>=zminpad))   &&  ( (j<TBIFilter.y_num) && (k< (zminpad+TBIFilter.z_num))) && (i>=xminpad) && (i<xminpad+TBIFilter.x_num))
			//else if ( ((j<pb.y_num) && (k>=zminpad))   &&  ( (j<pb.y_num) && (k< (zminpad+pb.z_num))) )
				{
				//printf("second else if\n");
				counter++;
				//pb_Rando.mednum[k*pb_Rando.x_num*pb_Rando.y_num + j*pb_Rando.x_num +i] = pb.mednum[(k-zminpad)*pb.x_num*pb.y_num + j*pb.x_num +i ];
				if (TBIFilter.mednum[(k-zminpad)*TBIFilter.x_num*TBIFilter.y_num + j*TBIFilter.x_num + i-xminpad ] == 1)
				  TBI.mednum[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] = 1;
				else
				  TBI.mednum[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] = pb_Rando.num_mat-1 + TBIFilter.mednum[(k-zminpad)*TBIFilter.x_num*TBIFilter.y_num + j*TBIFilter.x_num + i-xminpad ];
				}
			//else if ( ((k<zminpad) && (j<pb.y_num)) ||  ( (k >= (zminpad+pb.z_num)) && (j<pb.y_num) ) )
			else if ( ((k<zminpad) && (j<TBIFilter.y_num)) ||  ( (k >= (zminpad+TBIFilter.z_num)) && (j<TBIFilter.y_num) ) )
				{
				//printf("first else if\n");
				//printf("index=%d\n",k*Rando.x_num*Rando.y_num + j*Rando.x_num +i);
				TBI.mednum[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] = 1;
				}
			else if ( ((k>=zminpad) && (i<xminpad) && (k<zminpad+TBIFilter.z_num) && (j<TBIFilter.y_num)) ||  ((k>=zminpad) && (i>=xminpad+TBIFilter.x_num) && (k<zminpad+TBIFilter.z_num) && (j<TBIFilter.y_num)))
				{	
				TBI.mednum[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] = 1;
				}
			else  
				TBI.mednum[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] =  pb_Rando.mednum[k*pb_Rando.x_num*pb_Rando.y_num + (j-TBIFilter.y_num-1)*pb_Rando.x_num +i];
			//counter+=1;
			//printf("counter=%d\n",counter);
		}


//  printf("counter=%d\n",counter);

  printf("fusing densities\n");

  TBI.densval = (float *)calloc(TBI.x_num*TBI.y_num*TBI.z_num,sizeof(float));
  if(pb_Rando.densval == NULL)
  {
     printf("\n ERROR: Allocating Memory for Float Array");
     return(FAIL);
  }
  // now fusing densities
printf("\n counter=%d\n",counter);
  for(int k=0;k<TBI.z_num;k++)//k<pb_Rando.z_num;k++)
//int k=0;
	for(int j=0;j<TBI.y_num;j++)
		for(int i=0;i<TBI.x_num;i++)
		{
			//printf(" i=%d j=%d k=%d \n",i,j,k);
			if( j == TBIFilter.y_num)
				//pb_Rando.mednum[k*pb_Rando.x_num*pb_Rando.y_num + j*pb_Rando.x_num +i] = 1;
				TBI.densval[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] = air_density;

			else if ( ((j<TBIFilter.y_num) && (k>=zminpad))   &&  ( (j<TBIFilter.y_num) && (k< (zminpad+TBIFilter.z_num))) && (i>=xminpad) && (i<xminpad+TBIFilter.x_num))
			//else if ( ((j<pb.y_num) && (k>=zminpad))   &&  ( (j<pb.y_num) && (k< (zminpad+pb.z_num))) )
				{
				//printf("second else if\n");
				counter++;
				//pb_Rando.mednum[k*pb_Rando.x_num*pb_Rando.y_num + j*pb_Rando.x_num +i] = pb.mednum[(k-zminpad)*pb.x_num*pb.y_num + j*pb.x_num +i ];

				TBI.densval[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] =  TBIFilter.densval[(k-zminpad)*TBIFilter.x_num*TBIFilter.y_num + j*TBIFilter.x_num + i-xminpad ];
				}
			//else if ( ((k<zminpad) && (j<pb.y_num)) ||  ( (k >= (zminpad+pb.z_num)) && (j<pb.y_num) ) )
			else if ( ((k<zminpad) && (j<TBIFilter.y_num)) ||  ( (k >= (zminpad+TBIFilter.z_num)) && (j<TBIFilter.y_num) ) )
				{
				//printf("first else if\n");
				//printf("index=%d\n",k*Rando.x_num*Rando.y_num + j*Rando.x_num +i);
				TBI.densval[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] = air_density;
				}
			else if ( ((k>=zminpad) && (i<xminpad) && (k<zminpad+TBIFilter.z_num) && (j<TBIFilter.y_num)) ||  ((k>=zminpad) && (i>=xminpad+TBIFilter.x_num) && (k<zminpad+TBIFilter.z_num) && (j<TBIFilter.y_num)))
				{	
				TBI.densval[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] = air_density;
				}
			else  
				TBI.densval[k*TBI.x_num*TBI.y_num + j*TBI.x_num +i] =  pb_Rando.densval[k*pb_Rando.x_num*pb_Rando.y_num + (j-TBIFilter.y_num-1)*pb_Rando.x_num +i];
			//counter+=1;
			//printf("counter=%d\n",counter);
		}
  printf("\n Writing information into %s", outputegsphantFileName);
  if(write_phant(outputegsphantFileName, &TBI) != OK)      // also defined in phantomStructure.h warning &phantom for pointer
  {
     printf("\n ERROR: Writting EGS4 Phantom File %s\n", outputegsphantFileName); return(FAIL);
  } 
  printf("\n");
return(OK);
}
